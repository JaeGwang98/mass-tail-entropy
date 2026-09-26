"""E1 — POPE baseline accuracy as a function of H.

If extreme H (over-concentration or over-spread) is a hallucination signal,
baseline accuracy should drop near H≈0 and H≈1 and peak in the middle. We
plot a U-shaped (or J-shaped) curve per split + an aggregate panel.

Reads:  results/sbc_h_baseline_{random,popular,adversarial}.jsonl
Writes: results/sbc_h_vs_hallucination.png
        results/sbc_h_vs_hallucination_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TAU_MID = 0.5
BINS = np.linspace(0, 1, 11)            # 10 bins of width 0.1


def bin_acc(rows):
    """Return (bin_centers, accuracies, counts) — uses bins of width 0.1.
    Rows where H is None (fallback / <2 segments) are dropped."""
    rows = [r for r in rows if r.get("H") is not None]
    H = np.array([r["H"] for r in rows])
    correct = np.array([r["baseline_correct"] for r in rows]).astype(int)
    centers = 0.5 * (BINS[:-1] + BINS[1:])
    accs = np.full(len(centers), np.nan)
    cnts = np.zeros(len(centers), dtype=int)
    for j in range(len(centers)):
        m = (H >= BINS[j]) & (H < BINS[j + 1])
        if j == len(centers) - 1:        # last bin includes H=1
            m = (H >= BINS[j]) & (H <= BINS[j + 1])
        cnts[j] = m.sum()
        if cnts[j] > 0:
            accs[j] = correct[m].mean()
    return centers, accs, cnts


def plot_panel(ax, centers, accs, cnts, title, color="#1f77b4"):
    ok = cnts > 0
    ax.plot(centers[ok], accs[ok], "-o", color=color, linewidth=2,
            markersize=8, markeredgecolor='black', markeredgewidth=0.6)
    # error bars (Wald CI for each bin)
    p = np.where(ok, accs, np.nan); n = cnts.astype(float)
    se = np.sqrt(np.where(ok, p * (1 - p) / np.maximum(n, 1), 0))
    ax.errorbar(centers[ok], accs[ok], yerr=1.96 * se[ok], fmt='none',
                ecolor='gray', capsize=3, linewidth=1)
    # bin-count annotations
    for c, a, n_ in zip(centers, accs, cnts):
        if n_ > 0:
            ax.text(c, a + 0.02, f"n={n_}", ha='center', fontsize=7.5,
                    color='dimgray')
    ax.axvline(TAU_MID, color='red', linestyle='--', linewidth=1.2,
               label=f"τ_mid = {TAU_MID}")
    # split into MSB regime (H<τ) vs PMI candidate (H≥τ) shading
    ax.axvspan(0, TAU_MID, color="#1f77b4", alpha=0.07)
    ax.axvspan(TAU_MID, 1, color="#2ca02c", alpha=0.07)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"normalized entropy $H(\mathrm{softmax}(\phi))$")
    ax.set_ylabel("baseline (greedy) accuracy")
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower center')
    ax.grid(alpha=0.3)


def main():
    rows_by_split = {}
    all_rows = []
    for split in ("random", "popular", "adversarial"):
        p = ROOT / "results" / f"sbc_h_baseline_{split}.jsonl"
        if not p.exists():
            print(f"  ! missing {p}"); continue
        rows = [json.loads(l) for l in open(p)]
        rows = [r for r in rows if r.get("H") is not None]
        rows_by_split[split] = rows
        all_rows.extend(rows)
    if not rows_by_split:
        raise SystemExit("no input data — run baseline_on_h_scan.py first")

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), sharey=True)
    colors = {"random": "#1f77b4", "popular": "#ff7f0e",
              "adversarial": "#d62728"}
    summary = {}
    for ax, (split, rows) in zip(axes[:3], rows_by_split.items()):
        c, a, n = bin_acc(rows)
        plot_panel(ax, c, a, n, f"POPE / {split}  (N={len(rows)})",
                   color=colors[split])
        # buckets for summary
        H = np.array([r["H"] for r in rows])
        cor = np.array([r["baseline_correct"] for r in rows])
        lo_m = H < 0.2; mid_m = (H >= 0.2) & (H < 0.8); hi_m = H >= 0.8
        summary[split] = {
            "n_total": len(rows),
            "overall_baseline_acc": float(cor.mean()),
            "H<0.2": {"n": int(lo_m.sum()),
                      "baseline_acc": float(cor[lo_m].mean()) if lo_m.sum() else None},
            "0.2≤H<0.8": {"n": int(mid_m.sum()),
                          "baseline_acc": float(cor[mid_m].mean()) if mid_m.sum() else None},
            "H≥0.8": {"n": int(hi_m.sum()),
                      "baseline_acc": float(cor[hi_m].mean()) if hi_m.sum() else None},
        }
    c, a, n = bin_acc(all_rows)
    plot_panel(axes[3], c, a, n, f"POPE (all splits)  N={len(all_rows)}",
               color="#2ca02c")
    summary["all"] = {
        "n_total": len(all_rows),
        "overall_baseline_acc":
            float(np.mean([r["baseline_correct"] for r in all_rows]))}

    fig.suptitle("E1 — baseline greedy accuracy vs SHAP-entropy $H$ on POPE   "
                 "(left of τ_mid: MSB regime · right: PMI candidate)",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = ROOT / "results" / "sbc_h_vs_hallucination.png"
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f"✓ saved {out}")

    (ROOT / "results" / "sbc_h_vs_hallucination_summary.json").write_text(
        json.dumps(summary, indent=2))
    print("\n=== bucketed baseline accuracy ===")
    for split, s in summary.items():
        if split == "all":
            print(f"  all-splits: N={s['n_total']}  acc={s['overall_baseline_acc']:.3f}")
            continue
        print(f"  {split:13s}  overall={s['overall_baseline_acc']:.3f}  "
              f"H<0.2: {s['H<0.2']['baseline_acc']} ({s['H<0.2']['n']})  "
              f"mid: {s['0.2≤H<0.8']['baseline_acc']} ({s['0.2≤H<0.8']['n']})  "
              f"H≥0.8: {s['H≥0.8']['baseline_acc']} ({s['H≥0.8']['n']})")


if __name__ == "__main__":
    main()
