r"""Percentile bootstrap CIs for CHAIR_I (image-level) and POPE F1 (question-
level per split).

Context (paper/camera_ready.tex App. B, "Bootstrap confidence intervals",
~L1086-1095, and the tab:pope caption ~L1135-1137): the printed \sbc{} 95%
percentile bootstrap CIs are CHAIR_I (1000 image resamples) LLaVA
[12.01, 14.23], Qwen [4.96, 6.86]; POPE F1 (1000 resamples per split) LLaVA
[85.26, 86.89], Qwen [87.49, 88.92], with "both lower bounds above the
corresponding baseline F1". No script that produced these numbers exists in
this codebase. This script documents and implements one concrete,
defensible procedure, following the same conventions as
``scripts/mme_bootstrap_ci.py`` (which fixed an analogous problem for
tab:mme's CI, see its docstring / NUMCHECK_B.md / FIX2_stats.md):

  1. CHAIR_I point estimate: exact replica of ``run()`` in
     ``src/benchmarks/chair.py``: chair_i = (sum over images of
     len(hallucinated_objects)) / (sum over images of
     len(generated_objects)) -- i.e. a RATIO OF SUMS across the whole
     1000-image sample, not an average of per-image ratios.
  2. CHAIR resampling unit: IMAGES (each raw_ours_sbc.jsonl / raw_baseline
     record is one image's caption with its own hallucinated/generated
     object-mention counts). Percentile bootstrap resamples the 1000 images
     with replacement, B times, and recomputes the ratio-of-sums chair_i on
     each resampled multiset. B=1000 as the paper-reported figure, B=10000
     also reported to check stability. Fixed seed 0 (NumPy
     ``default_rng``).
  3. POPE point estimate: exact replica of ``score()`` in
     ``src/benchmarks/pope.py`` (tp/tn/fp/fn from gt/pred yes-no labels,
     f1 = 2*precision*recall/(precision+recall)), computed per split then
     averaged over the 3 official splits (random/popular/adversarial),
     matching how Table tab:pope's "F1" column is built from
     tab:pope-splits.
  4. POPE resampling unit: QUESTIONS. For each of the 3 splits
     independently, resample that split's ~3000 questions with replacement
     (same split size each draw), compute F1 on the resampled split, then
     average the 3 resampled-split F1s to get one bootstrap draw of the
     Table tab:pope F1. Repeat B times; report the [2.5, 97.5] percentile
     of the B averaged-F1 draws. B=1000 (paper), B=10000 also reported.
     Fixed seed 0.
  5. Distinctness: every resampled index (image or question) is used
     directly to index a plain float/bool array and summed -- there is no
     intermediate grouping by image_id/qid, so a duplicate draw of the same
     original item is never silently merged with itself (the bug class
     that an earlier version of mme_bootstrap_ci.py had, where duplicate
     image draws sharing one dict key collapsed together during
     dict-based acc+ grouping).

Usage:
    python scripts/chair_pope_bootstrap_ci.py

Reads only from results/ (read-only); writes nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"

# --- CHAIR run files -------------------------------------------------------
# Point estimates confirmed to reproduce printed values exactly:
#   LLaVA SBC chair_i=13.1407 (App.B/tab:chair: 13.14), Qwen SBC 5.8348 (5.83)
#   LLaVA Baseline 14.1272 (tab:chair Baseline row: 14.13)
#   Qwen  Baseline  8.1150 (tab:chair Baseline row: 8.12)
CHAIR_RUNS = {
    "LLaVA": {
        "SBC": RESULTS / "chair_sbc_v3_b18_margin05_llava7b" / "raw_ours_sbc.jsonl",
        "Baseline": RESULTS / "chair_full" / "raw_baseline.jsonl",
    },
    "Qwen": {
        "SBC": RESULTS / "chair_sbc_v3_b18_margin05_qwen2_5vl" / "raw_ours_sbc.jsonl",
        "Baseline": RESULTS / "chair_baseline_greedy_qwen2_5vl" / "raw_baseline.jsonl",
    },
}
EXPECTED_CHAIR_I = {"LLaVA": 13.14, "Qwen": 5.83}          # tab:chair \sbc{} row
EXPECTED_CHAIR_I_BASELINE = {"LLaVA": 14.13, "Qwen": 8.12}  # tab:chair Baseline row

# --- POPE run directories ---------------------------------------------------
# Point estimates confirmed to reproduce printed values exactly:
#   LLaVA SBC avg F1=86.0390 (tab:pope: 86.04), Qwen SBC avg F1=88.1730 (88.17)
#   LLaVA Baseline avg F1=84.0915 (tab:pope Baseline row: 84.09)
#   Qwen  Baseline avg F1=84.8630 (tab:pope Baseline row: 84.86)
POPE_SPLITS = ["random", "popular", "adversarial"]
POPE_RUNS = {
    "LLaVA": {
        "SBC": (RESULTS / "pope_sbc_v3_b18_margin05_llava7b", "raw_ours_sbc"),
        "Baseline": (RESULTS / "pope_full", "raw_baseline_greedy"),
    },
    "Qwen": {
        "SBC": (RESULTS / "pope_sbc_v3_b18_margin05_qwen2_5vl", "raw_ours_sbc"),
        "Baseline": (RESULTS / "pope_baseline_greedy_qwen2_5vl", "raw_baseline"),
    },
}
EXPECTED_POPE_F1 = {"LLaVA": 86.04, "Qwen": 88.17}           # tab:pope \sbc{} row
EXPECTED_POPE_F1_BASELINE = {"LLaVA": 84.09, "Qwen": 84.86}  # tab:pope Baseline row


# ---------------------------------------------------------------------------
# Shared I/O
# ---------------------------------------------------------------------------
def load_raw(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# CHAIR_I: ratio of sums, image-level bootstrap
# ---------------------------------------------------------------------------
def chair_i_point(records: list[dict]) -> float:
    """Exact replica of chair_i = n_hallu_words / n_total_words in
    src/benchmarks/chair.py's run(), as a percentage."""
    h = sum(len(r["hallucinated_objects"]) for r in records)
    t = sum(len(r["generated_objects"]) for r in records)
    return h / t * 100.0


def chair_bootstrap(records: list[dict], B: int, seed: int) -> tuple[np.ndarray, float, float]:
    """Percentile bootstrap: resample IMAGES with replacement, B draws,
    recompute the ratio-of-sums chair_i on each resampled multiset."""
    n = len(records)
    hallu = np.array([len(r["hallucinated_objects"]) for r in records], dtype=np.float64)
    total = np.array([len(r["generated_objects"]) for r in records], dtype=np.float64)
    rng = np.random.default_rng(seed)
    scores = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)  # resample images w/ replacement
        scores[b] = hallu[idx].sum() / total[idx].sum() * 100.0
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return scores, float(lo), float(hi)


# ---------------------------------------------------------------------------
# POPE F1: question-level bootstrap per split, averaged over splits
# ---------------------------------------------------------------------------
def _f1_from_bool_arrays(gt_yes: np.ndarray, pred_yes: np.ndarray) -> float:
    tp = int(np.sum(gt_yes & pred_yes))
    fp = int(np.sum((~gt_yes) & pred_yes))
    fn = int(np.sum(gt_yes & (~pred_yes)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2 * precision * recall / max(1e-12, precision + recall)


def pope_split_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    gt_yes = np.array([r["gt"].strip().lower() == "yes" for r in records])
    pred_yes = np.array([r["pred"].strip().lower() == "yes" for r in records])
    return gt_yes, pred_yes


def pope_avg_f1_point(split_records: dict[str, list[dict]]) -> float:
    f1s = []
    for split in POPE_SPLITS:
        gt_yes, pred_yes = pope_split_arrays(split_records[split])
        f1s.append(_f1_from_bool_arrays(gt_yes, pred_yes))
    return sum(f1s) / len(f1s) * 100.0


def pope_avg_f1_bootstrap(split_records: dict[str, list[dict]], B: int, seed: int
                           ) -> tuple[np.ndarray, float, float]:
    """Percentile bootstrap: for each split independently, resample that
    split's QUESTIONS with replacement (same split size each draw), compute
    F1 on the resampled split, then average the 3 resampled-split F1s. B
    draws total; report the [2.5, 97.5] percentile of the averaged-F1
    draws."""
    arrays = {split: pope_split_arrays(split_records[split]) for split in POPE_SPLITS}
    rng = np.random.default_rng(seed)
    avg_f1s = np.empty(B, dtype=np.float64)
    for b in range(B):
        f1s = []
        for split in POPE_SPLITS:
            gt_yes, pred_yes = arrays[split]
            n = len(gt_yes)
            idx = rng.integers(0, n, size=n)  # resample questions w/ replacement
            f1s.append(_f1_from_bool_arrays(gt_yes[idx], pred_yes[idx]))
        avg_f1s[b] = sum(f1s) / len(f1s) * 100.0
    lo, hi = np.percentile(avg_f1s, [2.5, 97.5])
    return avg_f1s, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("Step 1: point-estimate reproduction check")
    print("=" * 78)

    chair_raw: dict[str, dict[str, list[dict]]] = {}
    point_ok = True
    for backbone, methods in CHAIR_RUNS.items():
        chair_raw[backbone] = {}
        for method, path in methods.items():
            raw = load_raw(path)
            chair_raw[backbone][method] = raw
            score = chair_i_point(raw)
            tag = ""
            if method == "SBC":
                expected = EXPECTED_CHAIR_I[backbone]
            else:
                expected = EXPECTED_CHAIR_I_BASELINE[backbone]
            match = abs(score - expected) < 0.02
            point_ok &= match
            tag = f"  vs printed {expected}  -> {'MATCH' if match else 'MISMATCH'}"
            print(f"CHAIR {backbone:6s} {method:9s} n_images={len(raw):5d} "
                  f"chair_i={score:7.4f}{tag}")
    print()

    pope_raw: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for backbone, methods in POPE_RUNS.items():
        pope_raw[backbone] = {}
        for method, (dirpath, prefix) in methods.items():
            split_records = {}
            for split in POPE_SPLITS:
                split_records[split] = load_raw(dirpath / f"{prefix}_{split}_run0.jsonl")
            pope_raw[backbone][method] = split_records
            score = pope_avg_f1_point(split_records)
            if method == "SBC":
                expected = EXPECTED_POPE_F1[backbone]
            else:
                expected = EXPECTED_POPE_F1_BASELINE[backbone]
            match = abs(score - expected) < 0.02
            point_ok &= match
            ns = {s: len(split_records[s]) for s in POPE_SPLITS}
            tag = f"  vs printed {expected}  -> {'MATCH' if match else 'MISMATCH'}"
            print(f"POPE  {backbone:6s} {method:9s} n_per_split={ns} "
                  f"avg_f1={score:7.4f}{tag}")
    print()

    if not point_ok:
        print("WARNING: at least one point estimate did not reproduce -- CIs "
              "below would be computed on the wrong run directory. Aborting.")
        return
    print("All point estimates reproduce the printed CHAIR_I / POPE F1 values "
          "exactly. Proceeding to bootstrap.")
    print()

    print("=" * 78)
    print("Step 2: CHAIR_I image-level percentile bootstrap (seed=0)")
    print("=" * 78)
    seed = 0
    chair_ci: dict[str, dict[str, tuple[float, float]]] = {}
    for backbone in ["LLaVA", "Qwen"]:
        chair_ci[backbone] = {}
        for method in ["SBC", "Baseline"]:
            raw = chair_raw[backbone][method]
            point = chair_i_point(raw)
            for B in (1000, 10000):
                _, lo, hi = chair_bootstrap(raw, B=B, seed=seed)
                if B == 1000:
                    chair_ci[backbone][method] = (lo, hi)
                print(f"{backbone:6s} {method:9s} point={point:6.2f}  "
                      f"B={B:5d}  95% CI = [{lo:.2f}, {hi:.2f}]  "
                      f"width={hi - lo:.2f}")
        print()

    print("=" * 78)
    print("Step 3: POPE F1 (avg over splits) question-level percentile bootstrap (seed=0)")
    print("=" * 78)
    pope_ci: dict[str, dict[str, tuple[float, float]]] = {}
    for backbone in ["LLaVA", "Qwen"]:
        pope_ci[backbone] = {}
        for method in ["SBC", "Baseline"]:
            split_records = pope_raw[backbone][method]
            point = pope_avg_f1_point(split_records)
            for B in (1000, 10000):
                _, lo, hi = pope_avg_f1_bootstrap(split_records, B=B, seed=seed)
                if B == 1000:
                    pope_ci[backbone][method] = (lo, hi)
                print(f"{backbone:6s} {method:9s} point={point:6.2f}  "
                      f"B={B:5d}  95% CI = [{lo:.2f}, {hi:.2f}]  "
                      f"width={hi - lo:.2f}")
        print()

    print("=" * 78)
    print("Step 4: seed sensitivity check (SBC only, B=1000)")
    print("=" * 78)
    for backbone in ["LLaVA", "Qwen"]:
        for s in (0, 1, 2):
            _, lo, hi = chair_bootstrap(chair_raw[backbone]["SBC"], B=1000, seed=s)
            print(f"CHAIR {backbone:6s} SBC  seed={s}  95% CI = [{lo:.2f}, {hi:.2f}]")
        for s in (0, 1, 2):
            _, lo, hi = pope_avg_f1_bootstrap(pope_raw[backbone]["SBC"], B=1000, seed=s)
            print(f"POPE  {backbone:6s} SBC  seed={s}  95% CI = [{lo:.2f}, {hi:.2f}]")
    print()

    print("=" * 78)
    print("Step 5: comparison to baseline point estimates (B=1000 CIs above)")
    print("=" * 78)
    for backbone in ["LLaVA", "Qwen"]:
        base_chair = EXPECTED_CHAIR_I_BASELINE[backbone]
        sbc_lo, sbc_hi = chair_ci[backbone]["SBC"]
        contains = sbc_lo <= base_chair <= sbc_hi
        print(f"CHAIR {backbone:6s}: SBC 95% CI = [{sbc_lo:.2f}, {sbc_hi:.2f}], "
              f"baseline point = {base_chair:.2f}  -> baseline "
              f"{'IS' if contains else 'is NOT'} inside the SBC CI")

        base_pope = EXPECTED_POPE_F1_BASELINE[backbone]
        p_lo, p_hi = pope_ci[backbone]["SBC"]
        exceeds = p_lo > base_pope
        print(f"POPE  {backbone:6s}: SBC 95% CI = [{p_lo:.2f}, {p_hi:.2f}], "
              f"baseline point = {base_pope:.2f}  -> SBC lower bound "
              f"{'EXCEEDS' if exceeds else 'does NOT exceed'} baseline F1")
    print()


if __name__ == "__main__":
    main()
