"""Scan CHAIR images and rank by "qualitative viz interestingness":
  - MSB route taken (H < tau_mid)
  - φ concentrated (low H) on a segment that is NOT the largest by area.
    → SHAP picks a non-obvious object, not the trivially big one.
  - baseline caption differs from SBC caption (so the visualization shows
    actual rewriting effect, not a no-op).

Outputs results/chair_viz_candidates.jsonl ranked best→worst, so we can
pick the top candidate as the qualitative figure.

Usage:  CUDA_VISIBLE_DEVICES=1 python scripts/find_chair_viz_case.py [N]
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
from src.models.llava_wrapper import LlavaWrapper                # noqa: E402
from src.utils.segmentation import PanopticSegmenter             # noqa: E402
from src.utils.common import load_config                         # noqa: E402
from src.decoding.baseline import greedy_decode                  # noqa: E402
from src.decoding.ours_sbc import (_lookahead_with_logp,         # noqa: E402
                                   _norm_entropy, _generate_msb)
from src.decoding.ours_msb import _shap_phis_batched, _topk_diverse  # noqa: E402
from src.benchmarks.chair import CHAIR_PROMPT, sample_image_ids  # noqa: E402

TAU_MID = 0.5
TOP_K = 2
BOOST = 1.5


def interestingness(row):
    """Higher = more interesting for visualization."""
    if row["route"] != "msb":
        return -1e9
    if not row["caption_changed"]:
        return -1e9                    # no visible effect → boring
    if row["H"] >= TAU_MID:
        return -1e9                    # not concentrated
    top1 = row["chosen"][0]            # index into segments (sorted by area desc)
    # we want top1 != 0 (not the biggest), and a small area frac
    if top1 == 0:
        return -1e9
    # base score: low H + small area of top-1 chosen segment
    return (TAU_MID - row["H"]) + (1.0 - row["areas"][top1])


@torch.no_grad()
def run_one(w, seg, image):
    enc = w.prepare_inputs(image, CHAIR_PROMPT)
    input_ids = enc["input_ids"]; pixel_v = enc["pixel_values"]
    attn = enc.get("attention_mask")
    segs = seg.segment(image, min_area_frac=0.01, max_segments=6)
    if len(segs) < 2:
        return None
    span, base_lp, *_ = _lookahead_with_logp(
        w, input_ids, pixel_v, attn, True, 8, max_steps=32)
    if not span:
        return None
    phis = _shap_phis_batched(w, image, segs, input_ids, pixel_v, attn, span,
                              base_lp=base_lp)
    H = _norm_entropy(phis)
    chosen = _topk_diverse(phis, segs, TOP_K, 0.5) if H < TAU_MID else None
    return {"segments": segs, "phis": phis, "H": float(H), "chosen": chosen,
            "span": span}


def main():
    n_scan = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    cfg = load_config()
    w = LlavaWrapper(model_name=cfg["model"]["name"],
                     dtype=getattr(torch, cfg["model"]["dtype"]),
                     attn_implementation=cfg["model"]["attn_implementation"])
    seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                            dtype=getattr(torch, cfg["model"]["dtype"]))
    chair_dir = ROOT / cfg["benchmarks"]["chair"]["image_dir"]
    seed = cfg["benchmarks"]["chair"]["seed"]

    rows = []
    for k, (cid, fn) in enumerate(sample_image_ids(chair_dir, n_scan, seed)):
        img = Image.open(chair_dir / fn).convert("RGB")
        st = run_one(w, seg, img)
        if st is None:
            continue
        if st["H"] >= TAU_MID or st["chosen"] is None:
            print(f"[{k+1}/{n_scan}] {fn}  H={st['H']:.3f} → PMI/flat, skipping",
                  flush=True)
            continue
        base = greedy_decode(w, img, CHAIR_PROMPT, max_new_tokens=128)
        enc = w.prepare_inputs(img, CHAIR_PROMPT)
        sbc = _generate_msb(w, enc["input_ids"], enc["pixel_values"],
                            enc.get("attention_mask"),
                            st["segments"], st["phis"], TOP_K, 0.5, BOOST, 128)
        if sbc is None:
            sbc = base
        row = {"image": fn, "cid": cid, "H": st["H"],
               "phis": [float(x) for x in st["phis"]],
               "areas": [float(s.area_frac) for s in st["segments"]],
               "labels": [s.label for s in st["segments"]],
               "chosen": list(st["chosen"]),
               "route": "msb",
               "span": st["span"],
               "baseline": base, "sbc": sbc,
               "caption_changed": base.strip() != sbc.strip()}
        row["score"] = interestingness(row)
        rows.append(row)
        top1 = row["chosen"][0]
        rank_by_area = sorted(range(len(row["areas"])),
                              key=lambda i: -row["areas"][i]).index(top1) + 1
        print(f"[{k+1}/{n_scan}] {fn}  H={row['H']:.3f}  "
              f"chosen={row['chosen']}  top1_area_rank={rank_by_area}/"
              f"{len(row['areas'])}  area={row['areas'][top1]:.3f}  "
              f"changed={row['caption_changed']}  score={row['score']:.3f}",
              flush=True)

    rows.sort(key=lambda r: -r["score"])
    out = ROOT / "results" / "chair_viz_candidates.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✓ wrote {out}  ({len(rows)} rows)")
    print("\n=== top-5 candidates ===")
    for r in rows[:5]:
        print(f"  {r['image']}  H={r['H']:.3f}  chosen={r['chosen']}  "
              f"labels[top1]='{r['labels'][r['chosen'][0]]}'  "
              f"score={r['score']:.3f}")


if __name__ == "__main__":
    main()
