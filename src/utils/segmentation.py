"""Mask2Former panoptic segmentation wrapper for the LOO-attribution pipeline.

The wrapper returns a list of binary masks (one per accepted segment) at the
*original* image resolution.  Segments smaller than ``min_area_frac`` of the
total image area are discarded; at most ``max_segments`` are kept (sorted by
area, descending).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import (AutoImageProcessor,
                          Mask2FormerForUniversalSegmentation)


@dataclass
class Segment:
    mask: np.ndarray   # bool array, shape (H, W) — True = segment pixels
    area_frac: float   # fraction of total image pixels
    label_id: int = -1
    label: str = ""    # class name from id2label


class PanopticSegmenter:
    def __init__(self, model_name: str = "facebook/mask2former-swin-large-coco-panoptic",
                 device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device).eval()

    @torch.no_grad()
    def segment(self, image: Image.Image, min_area_frac: float = 0.01,
                max_segments: int = 6) -> List[Segment]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(self.dtype)
        outputs = self.model(**inputs)

        # post_process_panoptic_segmentation gives us a single label map plus
        # per-segment metadata, all at the (target_size) resolution.
        target_size = (image.height, image.width)
        result = self.processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[target_size]
        )[0]
        seg_map = result["segmentation"].cpu().numpy()      # (H, W) int
        seg_info = result["segments_info"]                  # list[dict]

        H, W = seg_map.shape
        total = float(H * W)
        id2label = self.model.config.id2label
        candidates: List[Segment] = []
        for info in seg_info:
            seg_id = info["id"]
            mask = (seg_map == seg_id)
            area_frac = float(mask.sum()) / total
            if area_frac >= min_area_frac:
                lid = int(info.get("label_id", -1))
                lname = id2label.get(lid, "") if lid >= 0 else ""
                candidates.append(Segment(mask=mask, area_frac=area_frac,
                                          label_id=lid, label=lname))
        candidates.sort(key=lambda s: s.area_frac, reverse=True)
        return candidates[:max_segments]


class SAMSegmenter:
    """Class-agnostic SAM (ViT-B) automatic mask generation with the same
    ``segment()`` interface as :class:`PanopticSegmenter`.

    Masks are greedily made disjoint (largest first; each mask minus the
    union of already-accepted ones) so the LOO occlusion does not
    double-count pixels, matching the disjoint panoptic partition."""

    def __init__(self, model_name: str = "facebook/sam-vit-base",
                 device: str = "cuda", dtype: torch.dtype = torch.float32,
                 points_per_side: int = 16):
        # fp32 regardless of the VLM dtype: the mask-generation pipeline's
        # torchvision NMS raises "dets should have the same type as scores"
        # under fp16, and SAM-ViT-B is small enough (~0.4 GB) for fp32.
        from transformers import pipeline
        self.pipe = pipeline("mask-generation", model=model_name,
                             device=0 if device == "cuda" else -1,
                             torch_dtype=torch.float32)
        self.points_per_side = points_per_side

    @torch.no_grad()
    def segment(self, image: Image.Image, min_area_frac: float = 0.01,
                max_segments: int = 6) -> List[Segment]:
        out = self.pipe(image, points_per_side=self.points_per_side,
                        points_per_batch=64)
        masks = [np.asarray(m, dtype=bool) for m in out["masks"]]
        masks.sort(key=lambda m: m.sum(), reverse=True)
        total = float(image.height * image.width)
        taken = np.zeros((image.height, image.width), dtype=bool)
        segs: List[Segment] = []
        for m in masks:
            if m.shape != taken.shape:
                continue
            m2 = m & ~taken
            area_frac = float(m2.sum()) / total
            if area_frac >= min_area_frac:
                segs.append(Segment(mask=m2, area_frac=area_frac,
                                    label_id=-1, label="sam"))
                taken |= m2
            if len(segs) >= max_segments:
                break
        return segs


def mask_image_with_segment(image: Image.Image, segment_mask: np.ndarray) -> Image.Image:
    """Replace pixels inside ``segment_mask`` with the *mean color* of the
    original image (per the v3 spec: 'mean color fill')."""
    arr = np.asarray(image).copy()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    mean_color = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    arr[segment_mask] = mean_color.astype(arr.dtype)
    return Image.fromarray(arr)


def mask_image_with_segments(image: Image.Image,
                             segment_masks: "List[np.ndarray]",
                             fill: str = "mean") -> Image.Image:
    """Occlude the *union* of ``segment_masks``.  ``fill`` selects the occluder
    (for the fill-robustness study, §C6):

      - ``mean``   : original image's mean color, once (paper default)
      - ``zero``   : black (0,0,0)
      - ``gaussian``: i.i.d. noise ~ N(per-channel mean, per-channel std), clipped
      - ``blur``   : the heavily Gaussian-blurred original image at those pixels

    This is the coalition-masking primitive for ExactSHAP (see
    ``2026-06-08/method_exactshap_fastshap.md`` §1.5[M2]/§3.4).  Calling
    ``mask_image_with_segment`` sequentially per segment re-computes the mean
    on the *already-partially-filled* image each time, so the fill value drifts
    with the number/order of masked segments — an order-dependent bias that
    grows with coalition size and would contaminate v(S).  Filling the union in
    one shot with the single original-image mean removes that bias and keeps the
    fill value identical across all 2^K coalitions.

    An empty ``segment_masks`` returns the original image unchanged (this is the
    grand-coalition case v(N) where nothing is hidden)."""
    arr = np.asarray(image).copy()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if not segment_masks:
        return Image.fromarray(arr)
    union = np.zeros(arr.shape[:2], dtype=bool)
    for m in segment_masks:
        union |= m
    flat = arr.reshape(-1, arr.shape[-1]).astype(np.float64)
    if fill == "mean":
        arr[union] = flat.mean(axis=0).astype(arr.dtype)
    elif fill == "zero":
        arr[union] = 0
    elif fill == "gaussian":
        mu = flat.mean(axis=0); sd = flat.std(axis=0)
        # deterministic per-image RNG (seeded by image mean) so v(S) is stable
        rng = np.random.default_rng(int(abs(mu.sum() * 1000)) % (2**32))
        noise = rng.normal(mu, sd + 1e-6, size=(int(union.sum()), arr.shape[-1]))
        arr[union] = np.clip(noise, 0, 255).astype(arr.dtype)
    elif fill == "blur":
        from PIL import ImageFilter
        blurred = np.asarray(
            Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=20)))
        arr[union] = blurred[union]
    else:
        raise ValueError(f"unknown fill: {fill}")
    return Image.fromarray(arr)
