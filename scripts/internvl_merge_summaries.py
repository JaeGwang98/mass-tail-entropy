"""Merge / summarise the InternVL3-8B decoding results.

1. POPE: the queue runs one split per invocation, so the runner's
   ``summary_<method>.json`` only holds the last split. The per-split copies
   ``summary_<method>_<split>.json`` are merged into ``summary_<method>.json``
   with the same {split: {...}} layout as a single ``--setting all`` run.
2. Writes logs/internvl_decoding_summary.md: MME / POPE / CHAIR numbers for
   baseline vs SBC plus the SBC route distribution (from raw jsonl).

  python scripts/internvl_merge_summaries.py [--pope-only]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SPLITS = ("random", "popular", "adversarial")
POPE = {"baseline_greedy": RES / "pope_baseline_greedy_internvl3_8b",
        "ours_sbc": RES / "pope_sbc_internvl3_8b"}
MME = {"baseline_greedy": RES / "mme_baseline_greedy_internvl3_8b",
       "ours_sbc": RES / "mme_sbc_internvl3_8b"}
CHAIR = {"baseline": RES / "chair_baseline_greedy_internvl3_8b",
         "ours_sbc": RES / "chair_sbc_internvl3_8b"}


def merge_pope():
    for method, d in POPE.items():
        parts = {s: d / f"summary_{method}_{s}.json" for s in SPLITS}
        have = {s: p for s, p in parts.items() if p.exists()}
        if not have:
            continue
        merged = {}
        for s, p in have.items():
            merged.update({k: v for k, v in json.loads(p.read_text()).items()
                           if k == s})
        if len(merged) == len(SPLITS):
            (d / f"summary_{method}.json").write_text(json.dumps(merged, indent=2))
            print(f"merged {method}: {d / f'summary_{method}.json'}")
        else:
            print(f"{method}: only {sorted(merged)} available; not merged")


def routes(path: Path):
    if not path.exists():
        return None
    c = Counter()
    for line in path.read_text().splitlines():
        if line.strip():
            c[json.loads(line).get("route")] += 1
    return dict(c)


def fmt_routes(r):
    if not r:
        return "-"
    n = sum(r.values())
    return ", ".join(f"{k} {v / n * 100:.1f}%" for k, v in sorted(
        r.items(), key=lambda kv: -kv[1]))


def report():
    L = ["# InternVL3-8B decoding results (auto-generated)", "",
         "Protocol: greedy; SBC = gate v3, top-k=2, b=1.8, alpha=1.0, "
         "beta=0.1, delta=0.5, tau_mid=0.5, Mask2Former K<=6, lookahead<=32; "
         "tiling disabled (single 448x448 view), bf16.", ""]
    L += ["## MME hallucination subset (240 q)", "",
          "| method | subset score | acc | F1 | existence | count | position | color | SBC routes |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m, d in MME.items():
        p = d / f"summary_{m}.json"
        if not p.exists():
            L.append(f"| {m} | (pending) |||||||| ")
            continue
        s = json.loads(p.read_text())
        pc = s["per_category"]
        L.append(f"| {m} | {s['subset_score']:.1f} | {s['overall_acc']*100:.2f} | "
                 f"{s['overall_f1']*100:.2f} | "
                 + " | ".join(f"{pc[c]['score']:.1f}" for c in
                              ("existence", "count", "position", "color"))
                 + f" | {fmt_routes(routes(d / f'raw_{m}.jsonl')) if m != 'baseline_greedy' else '-'} |")
    L += ["", "## POPE (3000 q per split)", "",
          "| method | split | acc | prec | rec | F1 | yes% | SBC routes |",
          "|---|---|---|---|---|---|---|---|"]
    for m, d in POPE.items():
        for s in SPLITS:
            p = d / f"summary_{m}_{s}.json"
            if not p.exists():
                L.append(f"| {m} | {s} | (pending) ||||||")
                continue
            a = json.loads(p.read_text())[s]["aggregate"]
            r = routes(d / f"raw_{m}_{s}_run0.jsonl") if m == "ours_sbc" else None
            L.append(f"| {m} | {s} | {a['accuracy']['mean']*100:.2f} | "
                     f"{a['precision']['mean']*100:.2f} | {a['recall']['mean']*100:.2f} | "
                     f"{a['f1']['mean']*100:.2f} | {a['yes_ratio']['mean']*100:.2f} | "
                     f"{fmt_routes(r)} |")
    L += ["", "## CHAIR (1000 images, max 512 new tokens)", "",
          "| method | CHAIR_s | CHAIR_i | recall | avg len | SBC routes |",
          "|---|---|---|---|---|---|"]
    for m, d in CHAIR.items():
        p = d / f"summary_{m}.json"
        if not p.exists():
            L.append(f"| {m} | (pending) |||||")
            continue
        s = json.loads(p.read_text())
        r = routes(d / f"raw_{m}.jsonl") if m == "ours_sbc" else None
        L.append(f"| {m} | {s['chair_s']*100:.1f} | {s['chair_i']*100:.2f} | "
                 f"{s['recall']*100:.2f} | {s['avg_len']:.1f} | {fmt_routes(r)} |")
    out = ROOT / "logs" / "internvl_decoding_summary.md"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pope-only", action="store_true")
    a = ap.parse_args()
    merge_pope()
    report()
