"""Multi-Segment Boost (M-SB) — gate-free SHAP-based decoder.

Idea
----
Hallucinations cluster in over-concentrated SHAP regimes (CHAIR ~70% in 과집중).
v4 boosts only top-1 SHAP segment; M-SB boosts top-K (K=2,3) so the model
cannot rely on a single visual cue.

Cost
----
- Lookahead SHAP: K forwards (same as v3)
- Generation: 1 forward / token (cheaper than v3's 2)

Note
----
Only IoU-based NMS is applied to enforce diversity in the top-K selection;
no class-name filtering is used (kept generalizable across segmenters).
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from PIL import Image

from ..models.llava_wrapper import LlavaWrapper
from ..utils.segmentation import (PanopticSegmenter, Segment,
                                  mask_image_with_segment)
from .ours_v3 import _greedy_lookahead, _logp_span_under_image
from .ours_v4 import (_segment_to_visual_token_indices,
                      _build_boost_mask, _build_step_boost_mask)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / max(union, 1.0)


def _topk_diverse(phis: np.ndarray, segments: Sequence[Segment],
                  k: int, iou_thresh: float = 0.5) -> List[int]:
    """Pick top-K segment indices by phi, with NMS-style IoU diversity."""
    order = np.argsort(-phis)
    chosen: List[int] = []
    for idx in order.tolist():
        if any(_iou(segments[idx].mask, segments[j].mask) >= iou_thresh
               for j in chosen):
            continue
        chosen.append(idx)
        if len(chosen) >= k:
            break
    return chosen


def _shap_phis(wrapper, image, segments, prompt_ids, pixel_values, attn_mask,
               span):
    """Compute LOO SHAP phi per segment (sequential — K+1 forwards)."""
    base_lp = _logp_span_under_image(wrapper, prompt_ids, pixel_values,
                                     attn_mask, span)
    phis = []
    for seg in segments:
        masked_img = mask_image_with_segment(image, seg.mask)
        enc_i = wrapper.processor.image_processor(images=masked_img,
                                                  return_tensors="pt")
        pix_i = enc_i["pixel_values"].to(pixel_values.device,
                                         dtype=pixel_values.dtype)
        lp_i = _logp_span_under_image(wrapper, prompt_ids, pix_i, attn_mask,
                                      span)
        phis.append(base_lp - lp_i)
    return np.asarray(phis, dtype=np.float64)


@torch.no_grad()
def _shap_phis_batched(wrapper, image, segments, prompt_ids, pixel_values,
                       attn_mask, span, max_batch: int = 8, base_lp=None):
    """LOO SHAP phi via leave-one-segment-out occlusion.

    The model-specific occlusion forward is delegated to
    ``wrapper.logp_spans(images, prompt_ids, span)`` so this stays
    architecture-agnostic (LLaVA batches on the pixel axis; Qwen2-VL loops
    per image because its visual tokens are a flat patch sequence).

    If ``base_lp`` (= log p(span | x, v), accumulated during the lookahead)
    is supplied, the original image is *not* re-scored here (K forwards
    instead of K+1); otherwise it is prepended so phi = lp(orig) - lp(occl).
    """
    if not span:
        return np.zeros(len(segments), dtype=np.float64)
    images = [] if base_lp is not None else [image]
    images += [mask_image_with_segment(image, seg.mask) for seg in segments]
    lp_all = wrapper.logp_spans(images, prompt_ids, list(span),
                                max_batch=max_batch)
    if base_lp is not None:
        return (float(base_lp) - lp_all).detach().cpu().numpy().astype(np.float64)
    return (lp_all[0] - lp_all[1:]).detach().cpu().numpy().astype(np.float64)


@torch.no_grad()
def ours_msb_decode(wrapper: LlavaWrapper, segmenter: PanopticSegmenter,
                    image: Image.Image, question: str,
                    max_new_tokens: int = 64,
                    boost_factor: float = 2.0,
                    top_k: int = 3,
                    lookahead: int = 8,
                    nms_iou: float = 0.5,
                    min_area_frac: float = 0.01,
                    max_segments: int = 6,
                    use_sentence_lookahead: bool = False,
                    sentence_lookahead_max: int = 32) -> str:
    """Multi-Segment Boost decoding.

    Args:
        boost_factor: gamma — multiplicative attention boost on chosen segments.
        top_k: number of segments to boost (K=1 reduces to v4).
        nms_iou: IoU threshold for NMS-based diversity in top-K selection.
        lookahead: number of tokens for SHAP measurement (model self-terminates
            on EOS, so a single value works for both POPE and CHAIR).
    """
    enc = wrapper.prepare_inputs(image, question)
    input_ids = enc["input_ids"]
    pixel_v = enc["pixel_values"]
    attn_mask = enc.get("attention_mask")
    seq_len = input_ids.shape[1]
    device, dtype = wrapper.device, wrapper.dtype
    eos = int(wrapper.tokenizer.eos_token_id)

    visual_pos = wrapper.visual_token_positions(input_ids)
    text_start = visual_pos[-1] + 1

    # Step 0: segmentation
    segments = segmenter.segment(image, min_area_frac=min_area_frac,
                                 max_segments=max_segments)
    if not segments:
        from .baseline import greedy_decode
        return greedy_decode(wrapper, image, question, max_new_tokens)

    # Step 1: lookahead
    if use_sentence_lookahead:
        from .ours_msb_rolling import _greedy_lookahead_until_period
        span = _greedy_lookahead_until_period(
            wrapper, input_ids, pixel_v, attn_mask,
            max_steps=sentence_lookahead_max)
    else:
        span = _greedy_lookahead(wrapper, input_ids, pixel_v, attn_mask,
                                 lookahead)
    if not span:
        from .baseline import greedy_decode
        return greedy_decode(wrapper, image, question, max_new_tokens)

    # Step 2: per-segment SHAP.  Use the wrapper-delegated occlusion forward
    # (_shap_phis_batched) rather than the raw wrapper.model() path so that
    # Qwen2.5-VL's image_grid_thw is threaded through — the sequential
    # _shap_phis built pixel_values without grid_thw and crashed in
    # rot_pos_emb.  Semantics are identical: phi = lp(orig) - lp(occl).
    phis = _shap_phis_batched(wrapper, image, segments, input_ids,
                              pixel_v, attn_mask, span)

    # Step 3: top-K with NMS diversity
    chosen_idx = _topk_diverse(phis, segments, top_k, nms_iou)

    # Collect visual token positions for all chosen segments
    grid = wrapper.visual_grid(input_ids)
    boost_pos: List[int] = []
    for ci in chosen_idx:
        boost_pos.extend(_segment_to_visual_token_indices(
            segments[ci].mask, visual_pos, grid=grid))
    boost_pos = sorted(set(boost_pos))
    if not boost_pos:
        from .baseline import greedy_decode
        return greedy_decode(wrapper, image, question, max_new_tokens)

    # Step 4: boosted prefill (single forward) + greedy generation
    boost_mask = _build_boost_mask(seq_len, boost_pos, text_start,
                                   boost_factor, device, dtype)
    out = wrapper.prefill(input_ids, pixel_v, boost_mask)
    logits = out.logits[:, -1, :]
    pkv = out.past_key_values

    generated: List[int] = []
    cur_len = seq_len
    for _ in range(max_new_tokens):
        nxt = logits.argmax(-1, keepdim=True)
        token_id = int(nxt.item())
        if token_id == eos:
            break
        generated.append(token_id)
        step_mask = _build_step_boost_mask(cur_len, boost_pos,
                                           boost_factor, device, dtype)
        logits, pkv = wrapper.decode_step(nxt, pkv, attention_mask=step_mask)
        logits = logits[:, -1, :]
        cur_len += 1
    return wrapper.tokenizer.decode(generated, skip_special_tokens=True)
