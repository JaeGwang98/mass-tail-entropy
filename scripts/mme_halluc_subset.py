"""MME rows of the mass-tail table from ``results/mme_halluc_<model>.json``.

The raw file holds every MME question that received an H value. The paper
uses the hallucination subset only (existence / count / position / color).
Over that subset this prints n, the dominant tail and its share, and the
percentage of errors (Err) and of correct outputs (Corr) that fall in that
tail. Tails are the outer bins of the five equal-width H bins.

Usage:
  python scripts/mme_halluc_subset.py results/mme_halluc_llava-1-5-7b-hf.json
"""
from __future__ import annotations

import json
import sys

SUBSET = ("existence", "count", "position", "color")
LOW, HIGH = 0.2, 0.8          # over-concentration: H < 0.2, over-spread: H >= 0.8


def tail_of(h: float) -> str:
    if h < LOW:
        return "over-conc"
    if h >= HIGH:
        return "over-spread"
    return "middle"


def main(path: str) -> None:
    rows = [r for r in json.load(open(path))["raw"]
            if r["H"] is not None and r["cat"] in SUBSET]
    n = len(rows)
    counts = {t: sum(tail_of(r["H"]) == t for r in rows)
              for t in ("over-conc", "over-spread")}
    dom = max(counts, key=counts.get)
    err = [r for r in rows if not r["correct"]]
    cor = [r for r in rows if r["correct"]]

    def pct(group):
        return 100.0 * sum(tail_of(r["H"]) == dom for r in group) / max(1, len(group))

    print(f"{path}: n={n} dominant={dom} "
          f"share={100.0 * counts[dom] / n:.1f}% ({counts[dom]}) "
          f"Err={pct(err):.1f} Corr={pct(cor):.1f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
