"""E2 — route → recovery rate analysis.

For each POPE question in the 600q H-scan subset, cross-reference:
  - H, route (PMI / MSB / fallback)            ← from sbc_h_baseline_*.jsonl
  - baseline_correct                           ← from sbc_h_baseline_*.jsonl
  - SBC v3 prediction (from full POPE B run)   ← from pope_sbc_full_B/raw_*

Among baseline-errors, compute recovery rate per route:
  recovery = #(SBC_correct & baseline_wrong) / #(baseline_wrong)

Expected: PMI route ≫ recovers POPE prior-driven errors (high-H regime);
MSB route handles the rare low/mid-H cases.

Output: results/e2_route_recovery.json + results/e2_route_recovery.png
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

POPE_FULL_B = ROOT / "results" / "pope_sbc_full_B"


def load_sbc_preds(split):
    """Index by (image, question) — the qid in raw_*.jsonl is sequential
    (1..3000) while the qid in sbc_h_baseline_*.jsonl is the np.random index,
    so the two qid spaces don't line up."""
    rows = [json.loads(l) for l in open(
        POPE_FULL_B / f"raw_ours_sbc_{split}_run0.jsonl")]
    return {(r["image"], r["question"]): r for r in rows}


def main():
    summary = {}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharey=True)
    all_rows = []
    for ax, split in zip(axes[:3], ("random", "popular", "adversarial")):
        h = [json.loads(l) for l in open(
            ROOT / "results" / f"sbc_h_baseline_{split}.jsonl")]
        sbc = load_sbc_preds(split)
        rows = []
        misses = 0
        for r in h:
            key = (r["image"], r["question"])
            if key not in sbc:
                misses += 1; continue
            sp = sbc[key]["pred"].strip().lower()
            gt = str(r["gt"]).strip().lower()
            rows.append({**r, "sbc_pred": sp, "sbc_correct": sp == gt})
        if misses:
            print(f"  ! {split}: {misses} rows missed in raw lookup", flush=True)
        all_rows.extend([{**r, "split": split} for r in rows])

        # bucket: by route
        by_route = defaultdict(list)
        for r in rows:
            if r.get("route"):
                by_route[r["route"]].append(r)
        order = ["msb", "pmi", "fallback"]
        present = [k for k in order if k in by_route]
        # confusion: among baseline-wrong, was SBC right?
        route_stats = {}
        for k in present:
            sub = by_route[k]
            wrong = [r for r in sub if not r["baseline_correct"]]
            right = [r for r in sub if r["baseline_correct"]]
            recovered = sum(1 for r in wrong if r["sbc_correct"])
            broken = sum(1 for r in right if not r["sbc_correct"])
            route_stats[k] = {
                "n": len(sub),
                "n_baseline_wrong": len(wrong),
                "n_baseline_right": len(right),
                "recovered_by_sbc": recovered,
                "broken_by_sbc": broken,
                "recovery_rate": recovered / max(1, len(wrong)),
                "break_rate": broken / max(1, len(right)),
            }
        summary[split] = route_stats

        # bar plot: recovery rate per route on baseline-wrong subset
        labels = [f"{k.upper()}\n(n_err={route_stats[k]['n_baseline_wrong']})"
                  for k in present]
        rec = [100 * route_stats[k]["recovery_rate"] for k in present]
        brk = [100 * route_stats[k]["break_rate"] for k in present]
        x = np.arange(len(present)); w = 0.36
        ax.bar(x - w/2, rec, w, color="#2ca02c", edgecolor='black',
               linewidth=0.5, label="SBC recovers (baseline-wrong → SBC-right)")
        ax.bar(x + w/2, brk, w, color="#d62728", edgecolor='black',
               linewidth=0.5, label="SBC breaks (baseline-right → SBC-wrong)")
        for xi, r, b in zip(x, rec, brk):
            ax.text(xi - w/2, r + 1.5, f"{r:.0f}%", ha='center', fontsize=9)
            ax.text(xi + w/2, b + 1.5, f"{b:.0f}%", ha='center', fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylim(0, max(40, max(rec + brk + [10]) * 1.25))
        ax.set_title(f"POPE / {split}", fontsize=11, fontweight='bold')
        ax.set_ylabel("rate (%)")
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    # combined panel: ALL splits aggregated
    by_route = defaultdict(list)
    for r in all_rows:
        if r.get("route"):
            by_route[r["route"]].append(r)
    order = ["msb", "pmi", "fallback"]
    present = [k for k in order if k in by_route]
    all_stats = {}
    for k in present:
        sub = by_route[k]
        wrong = [r for r in sub if not r["baseline_correct"]]
        right = [r for r in sub if r["baseline_correct"]]
        recovered = sum(1 for r in wrong if r["sbc_correct"])
        broken = sum(1 for r in right if not r["sbc_correct"])
        all_stats[k] = {
            "n": len(sub), "n_baseline_wrong": len(wrong),
            "recovered_by_sbc": recovered, "broken_by_sbc": broken,
            "recovery_rate": recovered / max(1, len(wrong)),
            "break_rate": broken / max(1, len(right))}
    labels = [f"{k.upper()}\n(n_err={all_stats[k]['n_baseline_wrong']})"
              for k in present]
    rec = [100 * all_stats[k]["recovery_rate"] for k in present]
    brk = [100 * all_stats[k]["break_rate"] for k in present]
    ax = axes[3]
    x = np.arange(len(present)); w = 0.36
    ax.bar(x - w/2, rec, w, color="#2ca02c", edgecolor='black',
           linewidth=0.5, label="SBC recovers")
    ax.bar(x + w/2, brk, w, color="#d62728", edgecolor='black',
           linewidth=0.5, label="SBC breaks")
    for xi, r, b in zip(x, rec, brk):
        ax.text(xi - w/2, r + 1.5, f"{r:.0f}%", ha='center', fontsize=9)
        ax.text(xi + w/2, b + 1.5, f"{b:.0f}%", ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, max(40, max(rec + brk + [10]) * 1.25))
    ax.set_title(f"POPE (all splits)", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    summary["all"] = all_stats

    fig.suptitle("E2 — SBC v3 recovery on baseline errors,  stratified by route",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = ROOT / "results" / "e2_route_recovery.png"
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f"✓ saved {out}")

    (ROOT / "results" / "e2_route_recovery.json").write_text(
        json.dumps(summary, indent=2))
    print("\n=== route recovery summary ===")
    for k in present:
        s = all_stats[k]
        print(f"  {k.upper():9s}: n={s['n']}  baseline-wrong={s['n_baseline_wrong']}  "
              f"recovered={s['recovered_by_sbc']} ({100*s['recovery_rate']:.1f}%)  "
              f"broken={s['broken_by_sbc']} ({100*s['break_rate']:.1f}%)")


if __name__ == "__main__":
    main()
