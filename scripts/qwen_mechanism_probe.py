"""Diagnose WHY SBC fails on Qwen2-VL.

Measures the SBC gate's internal signals (SHAP-entropy H, route, blank-match,
per-segment phi spread) on CHAIR + POPE samples for an arbitrary model, so we
can compare Qwen2-VL against LLaVA-7B's known bimodal signature
(CHAIR H_med~0.18 concentrated, POPE H_med~1.0 spread).

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/qwen_mechanism_probe.py \
      Qwen/Qwen2-VL-7B-Instruct 150
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import build_wrapper                               # noqa: E402
from src.utils.segmentation import PanopticSegmenter               # noqa: E402
from src.utils.common import load_config                           # noqa: E402
from src.decoding.ours_sbc import (_lookahead_with_logp,           # noqa: E402
                                   _blank_span_matches, _norm_entropy)
from src.decoding.ours_msb import _shap_phis_batched               # noqa: E402
from src.benchmarks.pope import POPE_QUESTION_SUFFIX               # noqa: E402
from src.benchmarks.chair import CHAIR_PROMPT, sample_image_ids    # noqa: E402

TAU = 0.5


@torch.no_grad()
def measure(w, seg, image, question):
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
    H = float(_norm_entropy(phis))
    rec = {"H": H, "K": len(segs),
           "phi_max": float(np.max(phis)), "phi_min": float(np.min(phis)),
           "phi_range": float(np.max(phis) - np.min(phis))}
    if H >= TAU:
        m, *_ = _blank_span_matches(w, iid, torch.zeros_like(pv), am, span,
                                    True, 8, max_steps=32)
        rec["route"] = "pmi" if m else "msb"
        rec["blank_match"] = bool(m)
    else:
        rec["route"] = "msb"
        rec["blank_match"] = None
    return rec


def summarize(tag, recs):
    Hs = np.array([r["H"] for r in recs])
    routes = {}
    for r in recs:
        routes[r["route"]] = routes.get(r["route"], 0) + 1
    pr = np.array([r["phi_range"] for r in recs])
    bm = [r for r in recs if r["blank_match"] is not None]
    bm_rate = (sum(1 for r in bm if r["blank_match"]) / len(bm)
               if bm else float("nan"))
    print(f"\n=== {tag}  (n={len(recs)}) ===")
    print(f"  H: mean={Hs.mean():.3f} median={np.median(Hs):.3f} "
          f"std={Hs.std():.3f}  [<{TAU}: {(Hs<TAU).mean()*100:.0f}%  "
          f">={TAU}: {(Hs>=TAU).mean()*100:.0f}%]")
    print(f"  H deciles: " +
          " ".join(f"{np.quantile(Hs,q):.2f}" for q in np.linspace(0, 1, 11)))
    print(f"  routes: {routes}")
    print(f"  phi_range: mean={pr.mean():.3f} median={np.median(pr):.3f}")
    print(f"  blank-match rate (H>=tau cases): {bm_rate:.2f}")
    return {"H_mean": float(Hs.mean()), "H_med": float(np.median(Hs)),
            "routes": routes, "phi_range_mean": float(pr.mean())}


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2-VL-7B-Instruct"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    cfg = load_config()
    print(f"model={model}  n={n}", flush=True)
    w = build_wrapper(model_name=model,
                      dtype=getattr(torch, cfg["model"]["dtype"]),
                      attn_implementation=cfg["model"]["attn_implementation"])
    seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                            dtype=getattr(torch, cfg["model"]["dtype"]))

    # CHAIR sample
    cdir = ROOT / cfg["benchmarks"]["chair"]["image_dir"]
    seed = cfg["benchmarks"]["chair"]["seed"]
    chair = []
    for cid, fn in sample_image_ids(cdir, n, seed):
        r = measure(w, seg, Image.open(cdir / fn).convert("RGB"),
                    CHAIR_PROMPT)
        if r:
            chair.append(r)
        if len(chair) % 25 == 0 and chair:
            print(f"  [CHAIR] {len(chair)}/{n}", flush=True)
    sc = summarize(f"{model}  CHAIR", chair)

    # POPE sample (random split)
    import json
    pdir = ROOT / cfg["benchmarks"]["pope"]["image_dir"]
    qs = [json.loads(l) for l in open(
        ROOT / cfg["benchmarks"]["pope"]["data_dir"] /
        "coco_pope_random.json")][:n]
    pope = []
    for q in qs:
        ip = pdir / q["image"]
        if not ip.exists():
            continue
        r = measure(w, seg, Image.open(ip).convert("RGB"),
                    q["text"] + POPE_QUESTION_SUFFIX)
        if r:
            pope.append(r)
        if len(pope) % 25 == 0 and pope:
            print(f"  [POPE] {len(pope)}/{n}", flush=True)
    sp = summarize(f"{model}  POPE", pope)

    print("\n=== vs LLaVA-1.5-7B reference (bimodal) ===")
    print("  LLaVA CHAIR: H_med~0.18 (concentrated, ~MSB)")
    print("  LLaVA POPE : H_med~1.00 (spread, ~PMI)")
    print(f"  {model} CHAIR H_med={sc['H_med']:.3f}  POPE H_med={sp['H_med']:.3f}")
    sep = abs(sp["H_med"] - sc["H_med"])
    print(f"  bimodal separation |POPE-CHAIR H_med| = {sep:.3f}  "
          f"(LLaVA ~0.82; smaller => gate has less to separate)")


if __name__ == "__main__":
    main()
