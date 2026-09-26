"""Plot the bimodal-calibration validation figure.

Inputs:
  results/sbc_route_dist_chair.json   (CHAIR 199 imgs — H values pre-recorded)
  results/sbc_h_dist_pope.json        (POPE per-split summary)
  results/sbc_h_dist_pope_{split}.jsonl (POPE per-question H values)

Output: results/sbc_h_distribution.png

Layout:
  Row 1: H histograms, τ_mid=0.5 marked
    (a) CHAIR  (b) POPE-random  (c) POPE-popular  (d) POPE-adversarial
  Row 2: Route share stacked bars per benchmark
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TAU_MID = 0.5

ROUTE_COLOR = {"msb": "#1f77b4", "pmi": "#2ca02c", "fallback": "#7f7f7f"}


def bucket_chair(Hs):
    """Re-bucket the CHAIR H values with current single-threshold rule.
    Note: CHAIR captions almost never blank-agree, so H≥τ_mid still routes
    to MSB. We mark those as 'msb_overspread' for diagnostic visibility."""
    routes = {"msb_concentrated": 0, "msb_overspread": 0, "pmi": 0,
              "fallback": 0}
    for h in Hs:
        if h < TAU_MID:
            routes["msb_concentrated"] += 1
        else:
            routes["msb_overspread"] += 1   # CHAIR: blank-disagree → MSB
    return routes


def plot_hist(ax, Hs, title, color="#1f77b4"):
    bins = np.linspace(0, 1, 21)
    counts, edges = np.histogram(Hs, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    left_mask = centers < TAU_MID
    ax.bar(centers[left_mask], counts[left_mask], width=0.045,
           color="#1f77b4", edgecolor='black', linewidth=0.4,
           label=f"H<{TAU_MID} (MSB)")
    ax.bar(centers[~left_mask], counts[~left_mask], width=0.045,
           color="#2ca02c", edgecolor='black', linewidth=0.4,
           label=f"H≥{TAU_MID} (PMI candidate)")
    ax.axvline(TAU_MID, color='red', linestyle='--', linewidth=1.4,
               label=f"τ_mid = {TAU_MID}")
    n = len(Hs)
    n_lo = int((np.array(Hs) < TAU_MID).sum())
    n_hi = n - n_lo
    ax.set_title(f"{title}  (N={n}, low={n_lo}/{100*n_lo/n:.0f}%, "
                 f"high={n_hi}/{100*n_hi/n:.0f}%)",
                 fontsize=10.5, fontweight='bold')
    ax.set_xlabel(r"normalized entropy $H(\mathrm{softmax}(\phi))$")
    ax.set_ylabel("# samples")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc='upper center')
    ax.grid(axis='y', alpha=0.3)


def plot_route_stack(ax, labels, route_data):
    """route_data: list of dicts with keys {msb, pmi, fallback} (counts)."""
    keys = ["msb", "pmi", "fallback"]
    pretty = {"msb": "MSB (over-concentrated)",
              "pmi": "PMI (over-spread + blank-agree)",
              "fallback": "fallback (greedy)"}
    bottom = np.zeros(len(labels))
    width = 0.6
    x = np.arange(len(labels))
    totals = [sum(d.get(k, 0) for k in keys) for d in route_data]
    for k in keys:
        vals = np.array([100 * d.get(k, 0) / max(1, t) for d, t in
                         zip(route_data, totals)])
        bars = ax.bar(x, vals, width=width, bottom=bottom,
                      color=ROUTE_COLOR[k], edgecolor='black', linewidth=0.4,
                      label=pretty[k])
        for b, v in zip(bars, vals):
            if v >= 4:
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_y() + b.get_height() / 2,
                        f"{v:.0f}%", ha='center', va='center',
                        color='white', fontsize=10, fontweight='bold')
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("route share (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8.5, loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.12))
    ax.set_title("SBC v3 route distribution per benchmark",
                 fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, t in enumerate(totals):
        ax.text(i, 102, f"N={t}", ha='center', fontsize=9)


def main():
    chair_path = ROOT / "results" / "sbc_route_dist_chair.json"
    pope_path = ROOT / "results" / "sbc_h_dist_pope.json"

    chair = json.loads(chair_path.read_text())
    chair_H = np.array(chair["H_values"])

    pope = json.loads(pope_path.read_text())
    pope_per_split_H = {}
    pope_per_split_routes = {}
    for split in ("random", "popular", "adversarial"):
        path = ROOT / "results" / f"sbc_h_dist_pope_{split}.jsonl"
        rows = [json.loads(l) for l in open(path)]
        pope_per_split_H[split] = np.array([r["H"] for r in rows if "H" in r])
        pope_per_split_routes[split] = pope["splits"][split]["routes"]

    # CHAIR routes: re-bucket with τ_mid=0.5; blank-agree assumed false for
    # captions (existing JSON shows pmi_fired=0 — verified empirically).
    chair_routes_raw = bucket_chair(chair_H)
    chair_routes = {
        "msb": chair_routes_raw["msb_concentrated"] +
               chair_routes_raw["msb_overspread"],
        "pmi": 0,
        "fallback": int(chair.get("fallback", 0)),
    }
    # Add fallback count back to MSB total so percentages sum cleanly on N
    n_chair_total = int(chair["n"]) + int(chair.get("fallback", 0))

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0], hspace=0.45,
                          wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    plot_hist(ax, chair_H, f"CHAIR  ({len(chair_H)} captions)")
    for i, split in enumerate(("random", "popular", "adversarial")):
        ax = fig.add_subplot(gs[0, i + 1])
        plot_hist(ax, pope_per_split_H[split],
                  f"POPE / {split}  ({len(pope_per_split_H[split])} q)")

    ax = fig.add_subplot(gs[1, :])
    labels = ["CHAIR", "POPE-random", "POPE-popular", "POPE-adversarial"]
    route_data = [chair_routes,
                  pope_per_split_routes["random"],
                  pope_per_split_routes["popular"],
                  pope_per_split_routes["adversarial"]]
    plot_route_stack(ax, labels, route_data)

    fig.suptitle("SBC v3 — bimodal calibration validation:  H distribution & "
                 "route assignment across benchmarks",
                 fontsize=13.5, fontweight='bold', y=0.995)
    plt.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = ROOT / "results" / "sbc_h_distribution.png"
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f"✓ saved {out}")

    # also print a brief text summary
    print("\n=== summary ===")
    print(f"CHAIR  N={len(chair_H)}  H̄={chair_H.mean():.3f}  "
          f"med={np.median(chair_H):.3f}  "
          f"<{TAU_MID}: {int((chair_H<TAU_MID).sum())}  "
          f"≥{TAU_MID}: {int((chair_H>=TAU_MID).sum())}")
    for split in ("random", "popular", "adversarial"):
        h = pope_per_split_H[split]
        print(f"POPE/{split}  N={len(h)}  H̄={h.mean():.3f}  "
              f"med={np.median(h):.3f}  "
              f"<{TAU_MID}: {int((h<TAU_MID).sum())}  "
              f"≥{TAU_MID}: {int((h>=TAU_MID).sum())}  "
              f"routes={pope_per_split_routes[split]}")


if __name__ == "__main__":
    main()
