"""Image-level percentile bootstrap CI for the MME hallucination-subset score.

Context (paper/_proposals/NUMCHECK_B.md §3.3, FIX2_stats.md item 2): the
printed caption of ``tab:mme`` gives SBC 95% bootstrap CIs of
LLaVA [556, 639] and Qwen [632, 697], but the exact resampling procedure
that produced them is not present anywhere in this codebase, and two
independent re-implementations (by different audit agents) could not
reproduce that interval. This script documents and implements one concrete,
defensible procedure so the number in the paper is reproducible going
forward:

  1. Point estimate: replicate ``score_mme()`` from ``src/benchmarks/mme.py``
     exactly (subtask score = 100 * (acc + acc+), summed over the four
     hallucination subtasks: existence / count / position / color).
  2. Resampling unit: MME asks two paired yes/no questions per image within
     each subtask; ``acc+`` requires BOTH to be correct, so the two rows for
     one image must move together under resampling. We therefore resample
     IMAGES (not individual questions), with replacement, independently
     within each of the 4 categories (stratified by category, matching how
     the score itself is computed per-category before summing) -- this
     keeps each image's question pair intact and preserves the per-category
     sample size (30 images/category, 240 rows total).
  3. Percentile bootstrap: for B resamples, recompute the subset score on
     the resampled dataset; report the [2.5, 97.5] percentile of the B
     scores as the 95% CI. Fixed seed 0 (NumPy ``default_rng``), B=1000 as
     the paper-reported figure, with B=10000 also reported to check
     stability.

Usage:
    python scripts/mme_bootstrap_ci.py

Reads only from results/ (read-only); writes nothing.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"

HALLUCINATION_SUBTASKS = ["existence", "count", "position", "color"]

# Run directories confirmed in NUMCHECK_B.md sec.3 (point estimates MATCH
# the printed tab:mme scores exactly): SBC = mme_sbc_v3_b18_margin05_*
# (byte-identical to mme_sbc_v3_b18_delta0_5_*), baseline = mme_baseline_greedy_*.
RUNS = {
    "LLaVA": {
        "SBC": RESULTS / "mme_sbc_v3_b18_margin05_llava7b" / "raw_ours_sbc.jsonl",
        "Baseline": RESULTS / "mme_baseline_greedy_llava7b" / "raw_baseline_greedy.jsonl",
    },
    "Qwen": {
        "SBC": RESULTS / "mme_sbc_v3_b18_margin05_qwen2_5vl" / "raw_ours_sbc.jsonl",
        "Baseline": RESULTS / "mme_baseline_greedy_qwen2_5vl" / "raw_baseline_greedy.jsonl",
    },
}

# Printed point estimates (tab:mme, line ~1190 of paper/camera_ready.tex).
EXPECTED_SBC_SCORE = {"LLaVA": 598.3, "Qwen": 663.3}


def load_raw(path: Path) -> list[dict]:
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return recs


def score_mme(raw: list[dict]) -> float:
    """Exact replica of score_mme()'s subset_score in src/benchmarks/mme.py.

    Groups by (category, qid) -- i.e. by image within category, since MME's
    qid field here is "{category}/{image_file}", shared by the two questions
    asked about that image -- and sums 100*(acc+acc+) over the four
    hallucination subtasks."""
    total = 0.0
    for cat in HALLUCINATION_SUBTASKS:
        rows = [r for r in raw if r["category"] == cat]
        if not rows:
            continue
        acc = sum(r["correct"] for r in rows) / len(rows)
        by_img = defaultdict(list)
        for r in rows:
            by_img[r["qid"]].append(r["correct"])
        acc_plus = sum(all(v) for v in by_img.values()) / max(len(by_img), 1)
        total += (acc + acc_plus) * 100.0
    return total


def images_by_category(raw: list[dict]) -> dict[str, list[str]]:
    """List of unique image qids per category, in first-seen order."""
    out: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set] = defaultdict(set)
    for r in raw:
        cat, qid = r["category"], r["qid"]
        if qid not in seen[cat]:
            seen[cat].add(qid)
            out[cat].append(qid)
    return out


def bootstrap_ci(raw: list[dict], B: int, seed: int) -> tuple[np.ndarray, float, float]:
    """Percentile bootstrap: resample images with replacement, stratified by
    category, keeping each image's question pair intact."""
    by_cat_imgs = images_by_category(raw)
    # index rows by (category, qid) -> list of row dicts, for O(1) lookup
    rows_by_img: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in raw:
        rows_by_img[(r["category"], r["qid"])].append(r)

    rng = np.random.default_rng(seed)
    scores = np.empty(B, dtype=np.float64)
    for b in range(B):
        resampled_raw: list[dict] = []
        for cat, imgs in by_cat_imgs.items():
            n = len(imgs)
            idx = rng.integers(0, n, size=n)  # resample images w/ replacement
            # Suffix qid with the draw index so repeated draws of the same
            # image stay distinct in score_mme()'s acc+ grouping.
            for j, i in enumerate(idx):
                resampled_raw.extend({**r, "qid": f"{r['qid']}#{j}"}
                                     for r in rows_by_img[(cat, imgs[i])])
        scores[b] = score_mme(resampled_raw)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return scores, float(lo), float(hi)


def main() -> None:
    print("=" * 78)
    print("Step 1: point-estimate reproduction check (score_mme on raw records)")
    print("=" * 78)
    point_ok = True
    all_raw: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for backbone, methods in RUNS.items():
        for method, path in methods.items():
            raw = load_raw(path)
            all_raw[backbone][method] = raw
            score = score_mme(raw)
            n_rows = len(raw)
            n_imgs = sum(len(v) for v in images_by_category(raw).values())
            tag = ""
            if method == "SBC":
                expected = EXPECTED_SBC_SCORE[backbone]
                match = abs(score - expected) < 0.05
                point_ok &= match
                tag = f"  vs printed {expected}  -> {'MATCH' if match else 'MISMATCH'}"
            print(f"{backbone:6s} {method:9s} n_rows={n_rows:3d} n_images={n_imgs:3d} "
                  f"subset_score={score:.3f}{tag}")
    print()
    if not point_ok:
        print("WARNING: point estimate did not reproduce -- CI below would be "
              "computed on the wrong run directory. Aborting bootstrap.")
        return
    print("Point estimates reproduce the printed tab:mme SBC scores exactly "
          "(598.3 LLaVA, 663.3 Qwen). Proceeding to bootstrap.")
    print()

    print("=" * 78)
    print("Step 2: image-level stratified percentile bootstrap (seed=0)")
    print("=" * 78)
    seed = 0
    for backbone in ["LLaVA", "Qwen"]:
        for method in ["SBC", "Baseline"]:
            raw = all_raw[backbone][method]
            point = score_mme(raw)
            for B in (1000, 10000):
                _, lo, hi = bootstrap_ci(raw, B=B, seed=seed)
                print(f"{backbone:6s} {method:9s} point={point:6.1f}  "
                      f"B={B:5d}  95% CI = [{lo:.1f}, {hi:.1f}]  "
                      f"width={hi - lo:.1f}")
        print()

    print("=" * 78)
    print("Step 3: seed sensitivity check (SBC only, B=1000)")
    print("=" * 78)
    for backbone in ["LLaVA", "Qwen"]:
        raw = all_raw[backbone]["SBC"]
        for s in (0, 1, 2):
            _, lo, hi = bootstrap_ci(raw, B=1000, seed=s)
            print(f"{backbone:6s} SBC  seed={s}  95% CI = [{lo:.1f}, {hi:.1f}]")


if __name__ == "__main__":
    main()
