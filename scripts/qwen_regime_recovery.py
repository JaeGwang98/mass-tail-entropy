"""Qwen2-VL POPE regime-stratified recovery for all methods.

Measures per-question SBC SHAP-entropy H for a POPE subset on Qwen2-VL,
then joins with the already-computed method prediction raws
(baseline / VCD / AIF / OPERA-G / SBC-fixed) to report, per H regime
(over-concentration H<0.5, mid, over-spread H>=0.8), how many baseline
errors each method recovers — exactly the LLaVA-7B table, for Qwen2-VL.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/qwen_regime_recovery.py 400
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import build_wrapper                               # noqa: E402
from src.utils.segmentation import PanopticSegmenter               # noqa: E402
from src.utils.common import load_config                           # noqa: E402
from src.decoding.ours_sbc import _lookahead_with_logp, _norm_entropy  # noqa: E402
from src.decoding.ours_msb import _shap_phis_batched               # noqa: E402
from src.benchmarks.pope import POPE_QUESTION_SUFFIX               # noqa: E402

MODEL = "Qwen/Qwen2-VL-7B-Instruct"


def key(split, image, question):
    return (split, image, question.split(" Please answer")[0].strip())


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


def load_preds(tmpl):
    M = {}
    for s in ("random", "popular", "adversarial"):
        p = ROOT / tmpl.format(s=s)
        if not p.exists():
            return None
        for l in open(p):
            r = json.loads(l)
            M[key(s, r["image"], r["question"])] = (
                str(r["pred"]).strip().lower().startswith("yes"),
                str(r["gt"]).strip().lower().startswith("yes"))
    return M


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    cfg = load_config()
    w = build_wrapper(model_name=MODEL,
                      dtype=getattr(torch, cfg["model"]["dtype"]),
                      attn_implementation=cfg["model"]["attn_implementation"])
    seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                            dtype=getattr(torch, cfg["model"]["dtype"]))
    pdir = ROOT / cfg["benchmarks"]["pope"]["image_dir"]
    ddir = ROOT / cfg["benchmarks"]["pope"]["data_dir"]

    Hmap = {}
    for s in ("random", "popular", "adversarial"):
        qs = [json.loads(l) for l in open(ddir / f"coco_pope_{s}.json")][:n_per]
        for i, q in enumerate(qs):
            ip = pdir / q["image"]
            if not ip.exists():
                continue
            H = measure_H(w, seg, Image.open(ip).convert("RGB"),
                          q["text"] + POPE_QUESTION_SUFFIX)
            if H is not None:
                Hmap[key(s, q["image"], q["text"])] = H
            if (i + 1) % 100 == 0:
                print(f"  [{s}] {i+1}/{len(qs)}  (H measured {len(Hmap)})",
                      flush=True)

    base = load_preds("results/pope_base_qwen2vl/raw_baseline_greedy_{s}_run0.jsonl")
    methods = {
        "VCD": "results/pope_vcd_qwen2vl/raw_vcd_greedy_{s}_run0.jsonl",
        "AIF": "results/pope_aif_qwen2vl/raw_aif_{s}_run0.jsonl",
        "OPERA-G": "results/pope_opera_qwen2vl/raw_opera_{s}_run0.jsonl",
        "SBC v3": "results/pope_sbc_qwen2vl_fixed/raw_ours_sbc_{s}_run0.jsonl",
    }
    bcorrect = {k: (base[k][0] == base[k][1]) for k in base}
    keys = [k for k in Hmap if k in bcorrect]
    print(f"\nQwen2-VL POPE — joined {len(keys)}q (H measured ∩ baseline)")
    print(f'{"method":9s}|{"과집중 H<0.5":>13s}|{"중간 .5-.8":>12s}'
          f'|{"과퍼짐 H>=.8":>13s}| {"F1":>6s} {"P":>5s} {"R":>5s}')
    print("-" * 70)
    for nm, tm in methods.items():
        M = load_preds(tm)
        if M is None:
            print(f"{nm:9s}|  (raw 없음)"); continue
        jk = [k for k in keys if k in M]

        def rec(lo, hi):
            bw = [k for k in jk if lo <= Hmap[k] < hi and not bcorrect[k]]
            return sum(1 for k in bw if M[k][0] == M[k][1]), len(bw)
        r1, n1 = rec(0, .5); r2, n2 = rec(.5, .8); r3, n3 = rec(.8, 1.01)
        tp = fp = fn = tn = 0
        for k in jk:
            py, gy = M[k]
            if py and gy: tp += 1
            elif py and not gy: fp += 1
            elif not py and gy: fn += 1
            else: tn += 1
        P = tp / max(1, tp + fp); Rc = tp / max(1, tp + fn)
        F = 2 * P * Rc / max(1e-9, P + Rc)

        def c(r, n): return f"{r}/{n}({100*r/max(1,n):.0f}%)"
        print(f"{nm:9s}|{c(r1,n1):>13s}|{c(r2,n2):>12s}|{c(r3,n3):>13s}"
              f"| {F*100:>6.2f} {P*100:>5.1f} {Rc*100:>5.1f}")
    out = ROOT / "results" / "qwen_pope_H.jsonl"
    with open(out, "w") as f:
        for k, h in Hmap.items():
            f.write(json.dumps({"split": k[0], "image": k[1],
                                "q": k[2], "H": h}) + "\n")
    print(f"\n✓ wrote {out} ({len(Hmap)} per-q H)")


if __name__ == "__main__":
    main()
