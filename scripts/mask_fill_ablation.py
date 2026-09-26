"""App H — mask-fill robustness ablation.

For a fixed model (LLaVA-1.5-7B) and a fixed POPE random subset, recompute the
SHAP-H distribution under three different occlusion fills:
  * mean   — mean-color fill (paper default, current `mask_image_with_segment`)
  * zero   — black (0,0,0) fill
  * gauss  — i.i.d. Gaussian noise (mean=image mean, std=image std)

The expected outcome (defending Feedback #2): the H distribution shape is
qualitatively the same across fills — i.e. the diagnostic's claim that VLM
inference mass concentrates in a single H-tail is not an artifact of the
mean-color masking choice.

Output: results/app_h_mask_fill.json with per-fill {H_mean, H_median,
5-bin histogram, dominant-tail share, n_valid}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# We monkey-patch BEFORE importing decoders so the swap propagates everywhere.
from src.utils import segmentation as _seg_mod  # noqa: E402

_ORIG_MASK = _seg_mod.mask_image_with_segment


def _mean_fill(image: Image.Image, segment_mask: np.ndarray) -> Image.Image:
    arr = np.asarray(image).copy()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    mc = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    arr[segment_mask] = mc.astype(arr.dtype)
    return Image.fromarray(arr)


def _zero_fill(image: Image.Image, segment_mask: np.ndarray) -> Image.Image:
    arr = np.asarray(image).copy()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr[segment_mask] = 0
    return Image.fromarray(arr)


def _gauss_fill(image: Image.Image, segment_mask: np.ndarray) -> Image.Image:
    arr = np.asarray(image).copy()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    pix = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
    mean = pix.mean(axis=0); std = pix.std(axis=0) + 1e-6
    rng = np.random.default_rng(0)
    n_pix = int(segment_mask.sum())
    noise = rng.normal(loc=mean, scale=std, size=(n_pix, arr.shape[-1]))
    arr[segment_mask] = np.clip(noise, 0, 255).astype(arr.dtype)
    return Image.fromarray(arr)


FILLS = {"mean": _mean_fill, "zero": _zero_fill, "gauss": _gauss_fill}


def install_fill(name: str) -> None:
    _seg_mod.mask_image_with_segment = FILLS[name]
    # Also patch the names re-exported from decoders that did
    # `from ..utils.segmentation import mask_image_with_segment`.
    from src.decoding import ours_msb as _msb
    _msb.mask_image_with_segment = FILLS[name]


# Import the heavy modules AFTER defining the swap helper but BEFORE running.
from src.models import build_wrapper                                # noqa: E402
from src.utils.segmentation import PanopticSegmenter                # noqa: E402
from src.utils.common import load_config                            # noqa: E402
from src.decoding.ours_sbc import _lookahead_with_logp, _norm_entropy  # noqa: E402
from src.decoding.ours_msb import _shap_phis_batched                # noqa: E402
from src.benchmarks.pope import POPE_QUESTION_SUFFIX                # noqa: E402


@torch.no_grad()
def measure_H(w, seg, image, question):
    enc = w.prepare_inputs(image, question)
    iid, pv = enc["input_ids"], enc["pixel_values"]
    am = enc.get("attention_mask")
    segs = seg.segment(image, min_area_frac=0.01, max_segments=6)
    if len(segs) < 2:
        return None
    span, *_ = _lookahead_with_logp(w, iid, pv, am, True, 8, max_steps=32)
    if not span:
        return None
    phis = _shap_phis_batched(w, image, segs, iid, pv, am, span)
    return float(_norm_entropy(phis))


def summarise(Hs):
    Hs = np.asarray([h for h in Hs if h is not None])
    if len(Hs) == 0:
        return {"n": 0}
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.001]
    hist, _ = np.histogram(Hs, bins=edges)
    return {
        "n": int(len(Hs)),
        "H_mean": float(Hs.mean()),
        "H_median": float(np.median(Hs)),
        "hist_5bin": [int(x) for x in hist],
        "dominant_tail": "over-spread" if hist[-1] >= hist[0] else "over-concentration",
        "dominant_share": float(max(hist[-1], hist[0]) / hist.sum()),
    }


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "llava-hf/llava-1.5-7b-hf"
    n_q   = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    out_p = ROOT / "results" / "app_h_mask_fill.json"

    cfg = load_config()
    print(f"[setup] model={model}  n_q={n_q}", flush=True)
    w = build_wrapper(model_name=model,
                      dtype=getattr(torch, cfg["model"]["dtype"]),
                      attn_implementation=cfg["model"]["attn_implementation"])
    seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                            dtype=getattr(torch, cfg["model"]["dtype"]))

    pb = cfg["benchmarks"]["pope"]
    pdir = ROOT / pb["image_dir"]
    all_qs = [json.loads(l) for l in open(
        ROOT / pb["data_dir"] / "coco_pope_random.json")]
    # Dedup by image — first question per unique image — so we don't waste
    # the budget on 6 near-identical lookups per image and we cover ~n_q
    # distinct scenes.
    seen = set(); items = []
    for q in all_qs:
        if q["image"] in seen: continue
        seen.add(q["image"])
        ip = pdir / q["image"]
        if ip.exists():
            items.append((ip, q["text"] + POPE_QUESTION_SUFFIX))
        if len(items) >= n_q:
            break
    print(f"[setup] usable unique-image POPE questions: {len(items)}", flush=True)

    results = {}
    raw = {}
    for fill_name in ("mean", "zero", "gauss"):
        install_fill(fill_name)
        print(f"\n[run] fill={fill_name}", flush=True)
        Hs = []
        for k, (ip, full) in enumerate(items):
            try:
                img = Image.open(ip).convert("RGB")
                Hs.append(measure_H(w, seg, img, full))
            except Exception as e:
                Hs.append(None)
                if k < 3:
                    print(f"  ! qid={k} err={e}", flush=True)
            if (k + 1) % 50 == 0:
                v = sum(1 for x in Hs if x is not None)
                print(f"  [{fill_name}] {k+1}/{len(items)}  valid={v}",
                      flush=True)
        results[fill_name] = summarise(Hs)
        raw[fill_name] = [None if h is None else round(h, 6) for h in Hs]
        print(f"  -> {results[fill_name]}", flush=True)

    out_p.write_text(json.dumps({
        "model": model, "n_requested": len(items),
        "summary": results, "H_per_fill": raw,
    }, indent=2))
    print(f"\n✓ wrote {out_p}")


if __name__ == "__main__":
    main()
