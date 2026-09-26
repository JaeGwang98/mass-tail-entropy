"""Model-agnostic H-bin vs hallucination table (E1+E3 equivalent).

For a given model, on a CHAIR sample and a POPE sample:
  - measure SBC SHAP-entropy H (lookahead -> LOO SHAP -> norm entropy)
  - CHAIR: baseline greedy caption -> per-caption CHAIR-I (CHAIR matcher)
  - POPE : baseline greedy yes/no -> correctness
then bin both by H (deciles) so we can compare e.g. Qwen2-VL to LLaVA-7B.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/h_vs_hallu_anymodel.py \
      Qwen/Qwen2-VL-7B-Instruct 300 600
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
from src.models import build_wrapper                               # noqa: E402
from src.utils.segmentation import PanopticSegmenter               # noqa: E402
from src.utils.common import load_config                           # noqa: E402
from src.decoding.baseline import greedy_decode                    # noqa: E402
from src.decoding.ours_sbc import _lookahead_with_logp, _norm_entropy  # noqa: E402
from src.decoding.ours_msb import _shap_phis_batched               # noqa: E402
from src.benchmarks.chair import (CHAIR, CHAIR_PROMPT,             # noqa: E402
                                  sample_image_ids)
from src.benchmarks.pope import POPE_QUESTION_SUFFIX, parse_yes_no  # noqa: E402

BINS = np.linspace(0, 1, 11)


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


def binstats(rows, key):
    H = np.array([r["H"] for r in rows])
    v = np.array([r[key] for r in rows], dtype=float)
    out = []
    for j in range(10):
        m = (H >= BINS[j]) & ((H < BINS[j + 1]) if j < 9 else (H <= 1.0))
        if m.sum() == 0:
            continue
        out.append((BINS[j], BINS[j + 1], int(m.sum()), float(v[m].mean())))
    return out, float(v.mean()), len(rows)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2-VL-7B-Instruct"
    n_chair = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    n_pope = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    cfg = load_config()
    print(f"model={model} n_chair={n_chair} n_pope={n_pope}", flush=True)
    w = build_wrapper(model_name=model,
                      dtype=getattr(torch, cfg["model"]["dtype"]),
                      attn_implementation=cfg["model"]["attn_implementation"])
    seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                            dtype=getattr(torch, cfg["model"]["dtype"]))

    # ---- CHAIR ----
    cc = cfg["benchmarks"]["chair"]
    cdir = ROOT / cc["image_dir"]; adir = ROOT / cc["annotations_dir"]
    chair = CHAIR(ROOT / cc["synonyms_file"])
    ids = sample_image_ids(cdir, n_chair, cc["seed"])
    gt = chair.build_gt(adir / "instances_val2014.json",
                       adir / "captions_val2014.json",
                       [c for c, _ in ids])
    crows = []
    for k, (cid, fn) in enumerate(ids):
        img = Image.open(cdir / fn).convert("RGB")
        H = measure_H(w, seg, img, CHAIR_PROMPT)
        if H is None:
            continue
        cap = greedy_decode(w, img, CHAIR_PROMPT,
                            max_new_tokens=cc["max_new_tokens"])
        _, nw = chair.caption_to_objects(cap)
        hall = [x for x in nw if x not in gt.get(cid, set())]
        crows.append({"H": H,
                      "ci": len(hall) / max(1, len(nw)),
                      "hs": 1.0 if hall else 0.0})
        if (k + 1) % 50 == 0:
            print(f"  [CHAIR] {k+1}/{n_chair}", flush=True)

    # ---- POPE ----
    pb = cfg["benchmarks"]["pope"]
    pdir = ROOT / pb["image_dir"]
    qs = [json.loads(l) for l in open(
        ROOT / pb["data_dir"] / "coco_pope_random.json")][:n_pope]
    prows = []
    for k, q in enumerate(qs):
        ip = pdir / q["image"]
        if not ip.exists():
            continue
        img = Image.open(ip).convert("RGB")
        full = q["text"] + POPE_QUESTION_SUFFIX
        H = measure_H(w, seg, img, full)
        if H is None:
            continue
        ans = parse_yes_no(greedy_decode(w, img, full, max_new_tokens=8))
        prows.append({"H": H,
                      "correct": 1.0 if ans == str(q["label"]).lower() else 0.0})
        if (k + 1) % 75 == 0:
            print(f"  [POPE] {k+1}/{n_pope}", flush=True)

    cb, cm, cn = binstats(crows, "ci")
    csb, csm, _ = binstats(crows, "hs")
    pb_, pm, pn = binstats(prows, "correct")
    print(f"\n=== {model} — CHAIR H-bin (N={cn}) ===")
    print(f"  {'Hbin':>10s}{'n':>6s}{'CHAIR-S%':>10s}{'CHAIR-I%':>10s}")
    cs_map = {(lo, hi): (n, v) for lo, hi, n, v in csb}
    for lo, hi, n, ci in cb:
        hs = cs_map.get((lo, hi), (n, 0))[1]
        print(f"  [{lo:.1f},{hi:.1f}){n:>6d}{hs*100:>10.1f}{ci*100:>10.1f}")
    print(f"  overall   {cn:>6d}{csm*100:>10.1f}{cm*100:>10.1f}")
    print(f"\n=== {model} — POPE H-bin baseline acc (N={pn}) ===")
    print(f"  {'Hbin':>10s}{'n':>6s}{'acc%':>8s}")
    for lo, hi, n, acc in pb_:
        print(f"  [{lo:.1f},{hi:.1f}){n:>6d}{acc*100:>8.1f}")
    print(f"  overall   {pn:>6d}{pm*100:>8.1f}")

    out = ROOT / "results" / ("h_vs_hallu_" +
          model.split("/")[-1].replace(".", "").lower() + ".json")
    out.write_text(json.dumps({"model": model,
        "chair_bins": cb, "chair_s_bins": csb, "chair_overall_ci": cm,
        "pope_bins": pb_, "pope_overall_acc": pm,
        "n_chair": cn, "n_pope": pn}, indent=2))
    print(f"\n✓ wrote {out}")


if __name__ == "__main__":
    main()
