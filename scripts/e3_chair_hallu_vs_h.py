"""E3 — CHAIR per-caption hallucination rate as a function of H.

For each of N CHAIR images:
  1. Measure SBC-style H (Mask2Former -> SHAP -> normalized entropy).
  2. Generate baseline greedy caption (full 128 tokens).
  3. Parse caption objects with the CHAIR matcher.
  4. Compute per-caption hallucination flag and per-caption CI ratio.

Then bin by H and plot:
  - hallucination rate    (% captions with ≥1 hallucinated obj) per H bin
  - per-caption CI ratio  (#hallu_objs / #total_objs) per H bin

Output: results/e3_chair_hallu_vs_h.{jsonl,json,png}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.llava_wrapper import LlavaWrapper                # noqa: E402
from src.utils.segmentation import PanopticSegmenter             # noqa: E402
from src.utils.common import load_config                         # noqa: E402
from src.decoding.baseline import greedy_decode                  # noqa: E402
from src.decoding.ours_sbc import (_lookahead_with_logp,         # noqa: E402
                                   _norm_entropy)
from src.decoding.ours_msb import _shap_phis_batched              # noqa: E402
from src.benchmarks.chair import (CHAIR, CHAIR_PROMPT,           # noqa: E402
                                  sample_image_ids)

TAU_MID = 0.5


@torch.no_grad()
def measure_H(wrapper, segmenter, image, question):
    enc = wrapper.prepare_inputs(image, question)
    iid = enc["input_ids"]; pv = enc["pixel_values"]; am = enc.get("attention_mask")
    segs = segmenter.segment(image, min_area_frac=0.01, max_segments=6)
    if len(segs) < 2:
        return None, len(segs)
    span, *_ = _lookahead_with_logp(wrapper, iid, pv, am, True, 8, max_steps=32)
    if not span:
        return None, len(segs)
    phis = _shap_phis_batched(wrapper, image, segs, iid, pv, am, span)
    H = _norm_entropy(phis)
    return float(H), len(segs)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    cfg = load_config()
    cc = cfg["benchmarks"]["chair"]
    img_dir = ROOT / cc["image_dir"]
    ann_dir = ROOT / cc["annotations_dir"]
    syn_path = ROOT / cc["synonyms_file"]

    print(f"loading models;  N={n};  seed={cc['seed']}", flush=True)
    w = LlavaWrapper(model_name=cfg["model"]["name"],
                     dtype=getattr(torch, cfg["model"]["dtype"]),
                     attn_implementation=cfg["model"]["attn_implementation"])
    seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                            dtype=getattr(torch, cfg["model"]["dtype"]))

    print("building CHAIR GT ...", flush=True)
    chair = CHAIR(syn_path)
    ids = sample_image_ids(img_dir, n, cc["seed"])
    image_ids = [cid for cid, _ in ids]
    gt = chair.build_gt(ann_dir / "instances_val2014.json",
                       ann_dir / "captions_val2014.json", image_ids)

    out_rows = []
    t0 = time.time()
    max_new = cc["max_new_tokens"]
    for k, (cid, fn) in enumerate(ids):
        img = Image.open(img_dir / fn).convert("RGB")
        H, K = measure_H(w, seg, img, CHAIR_PROMPT)
        cap = greedy_decode(w, img, CHAIR_PROMPT, max_new_tokens=max_new)
        words, node_words = chair.caption_to_objects(cap)
        gt_set = gt.get(cid, set())
        hallu = [x for x in node_words if x not in gt_set]
        row = {"cid": cid, "image": fn, "H": H, "K": K,
               "caption": cap, "objects": node_words,
               "hallucinated": hallu, "n_obj": len(node_words),
               "n_hallu": len(hallu),
               "ci_per_caption": len(hallu) / max(1, len(node_words)),
               "has_hallu": int(len(hallu) > 0)}
        out_rows.append(row)
        if (k + 1) % 20 == 0:
            done = [r for r in out_rows if r["H"] is not None]
            mean_H = np.mean([r["H"] for r in done]) if done else float("nan")
            print(f"  [{k+1}/{n}]  H̄={mean_H:.3f}  hallu_rate="
                  f"{np.mean([r['has_hallu'] for r in out_rows]):.2f}  "
                  f"avg_ci={np.mean([r['ci_per_caption'] for r in out_rows]):.3f}  "
                  f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)

    out_dir = ROOT / "results"
    out_jsonl = out_dir / "e3_chair_hallu_vs_h.jsonl"
    with open(out_jsonl, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✓ wrote {out_jsonl}", flush=True)

    # ---- bin by H and plot ----
    valid = [r for r in out_rows if r["H"] is not None]
    H = np.array([r["H"] for r in valid])
    has_h = np.array([r["has_hallu"] for r in valid])
    ci = np.array([r["ci_per_caption"] for r in valid])
    bins = np.linspace(0, 1, 11)
    centers = 0.5 * (bins[:-1] + bins[1:])
    rates_s = np.full(len(centers), np.nan)
    rates_i = np.full(len(centers), np.nan)
    cnts = np.zeros(len(centers), dtype=int)
    for j in range(len(centers)):
        m = (H >= bins[j]) & (H < bins[j + 1])
        if j == len(centers) - 1:
            m = (H >= bins[j]) & (H <= bins[j + 1])
        cnts[j] = m.sum()
        if cnts[j] > 0:
            rates_s[j] = has_h[m].mean()
            rates_i[j] = ci[m].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, vals, ylabel, title in [
            (axes[0], rates_s, "P(caption has ≥1 hallucinated obj)",
             "CHAIR-S  rate vs H"),
            (axes[1], rates_i, "mean per-caption CI = #hallu / #total",
             "CHAIR-I  rate vs H")]:
        ok = cnts > 0
        ax.plot(centers[ok], vals[ok] * (100 if "CI" not in ylabel else 1),
                "-o", color="#d62728", linewidth=2, markersize=8,
                markeredgecolor='black', markeredgewidth=0.6)
        for c, v, n_ in zip(centers, vals, cnts):
            if n_ > 0:
                ax.text(c, v * (100 if "CI" not in ylabel else 1) + (0.02 if "CI" in ylabel else 2),
                        f"n={n_}", ha='center', fontsize=8, color='dimgray')
        ax.axvline(TAU_MID, color='red', linestyle='--', linewidth=1.3,
                   label=f"τ_mid = {TAU_MID}")
        ax.axvspan(0, TAU_MID, color="#1f77b4", alpha=0.07)
        ax.axvspan(TAU_MID, 1, color="#2ca02c", alpha=0.07)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"normalized entropy $H(\mathrm{softmax}(\phi))$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f"E3 — CHAIR baseline-greedy hallucination vs H  "
                 f"(N={len(valid)} images)",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    png = out_dir / "e3_chair_hallu_vs_h.png"
    plt.savefig(png, dpi=140, bbox_inches='tight')
    print(f"✓ saved {png}")

    summary = {
        "n_total": len(out_rows), "n_valid_H": len(valid),
        "overall_hallu_rate": float(has_h.mean()),
        "overall_avg_ci": float(ci.mean()),
        "by_bin": [{"bin": [float(bins[j]), float(bins[j + 1])],
                    "n": int(cnts[j]),
                    "hallu_rate": float(rates_s[j]) if cnts[j] > 0 else None,
                    "avg_ci": float(rates_i[j]) if cnts[j] > 0 else None}
                   for j in range(len(centers))],
    }
    (out_dir / "e3_chair_hallu_vs_h.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nOverall: hallu_rate={has_h.mean():.3f}  avg_ci={ci.mean():.3f}")
    print("By H bin:")
    for j in range(len(centers)):
        if cnts[j] > 0:
            print(f"  H ∈ [{bins[j]:.1f},{bins[j+1]:.1f}): n={cnts[j]:3d}  "
                  f"hallu_rate={rates_s[j]:.3f}  avg_ci={rates_i[j]:.3f}")


if __name__ == "__main__":
    main()
