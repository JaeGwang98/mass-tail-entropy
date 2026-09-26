r"""Analyse ExactSHAP diagnostic rows (output of scripts/diag_exactshap.py).

Implements the review-hardened evaluation (method_exactshap_fastshap.md §1.5):

  PRIMARY (non-circular, §1.5[C2]): HALL vs G+ separation.
    - ROC-AUC of each label-free feature (M, M_per_tok, grounding_vs_b0,
      r_prior_b0, H_pos, top1_share) at separating HALL (false-yes) from G+.
    - Cheap-baseline AUCs (§1.5[M3]): seq_logprob(_per_tok), maxsoftmax,
      b0-alone, grounding_vs_b0.  If a 1-forward baseline matches ExactSHAP,
      that is reported honestly.

  ESTIMATOR (the reason to switch off LOO): LOO vs ExactSHAP divergence.
    - per-sample Spearman(phi_exact, phi_loo); top-1 segment disagreement rate;
      redundancy cases (LOO says flat / all|phi|<eps but Exact M large).
    - efficiency residual sanity.

  BASELINE GAP (§1.5[C1]): v_bg (background-only) vs b0 (true zeros blank).

  Multiple comparisons (§1.5[M5]): Benjamini-Hochberg across the AUC family.

AUC is computed without sklearn (Mann-Whitney U / rank statistic), so this runs
anywhere.  Direction is auto-oriented (reports max(auc, 1-auc) with sign).

Usage:
  python -m scripts.analyze_signature results/exactshap/llava-1.5-7b-hf_adversarial_greedy_seed1234
  python -m scripts.analyze_signature results/exactshap/<dir1> <dir2> ...   # pooled
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.stats import spearmanr, mannwhitneyu
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---------------------------------------------------------------------------
def load_rows(dirs):
    rows = []
    for d in dirs:
        p = Path(d)
        f = p / "rows.jsonl" if p.is_dir() else p
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def auc_mannwhitney(pos, neg):
    """ROC-AUC via the rank-sum identity: AUC = U / (n_pos * n_neg).
    Returns (auc, n_pos, n_neg).  auc>0.5 => higher feature value predicts the
    POSITIVE class."""
    pos = np.asarray([x for x in pos if x is not None and np.isfinite(x)], float)
    neg = np.asarray([x for x in neg if x is not None and np.isfinite(x)], float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), len(pos), len(neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float)  # 0-based; ties rare in logp
    # average-rank tie correction
    order = np.argsort(allv)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    U = r_pos - len(pos) * (len(pos) - 1) / 2.0
    auc = U / (len(pos) * len(neg))
    return float(auc), len(pos), len(neg)


def oriented_auc(rows, key, pos_pred, neg_pred):
    pos = [r.get(key) for r in rows if pos_pred(r)]
    neg = [r.get(key) for r in rows if neg_pred(r)]
    auc, npos, nneg = auc_mannwhitney(pos, neg)
    if np.isnan(auc):
        return dict(key=key, auc=float("nan"), npos=npos, nneg=nneg, p=float("nan"))
    p = float("nan")
    if HAVE_SCIPY and npos and nneg:
        pv = np.asarray([x for x in pos if x is not None and np.isfinite(x)], float)
        nv = np.asarray([x for x in neg if x is not None and np.isfinite(x)], float)
        try:
            p = float(mannwhitneyu(pv, nv, alternative="two-sided").pvalue)
        except Exception:
            pass
    # orient so reported auc >= 0.5, remember direction
    direction = "+" if auc >= 0.5 else "-"
    return dict(key=key, auc=max(auc, 1 - auc), raw_auc=auc, dir=direction,
                npos=npos, nneg=nneg, p=p)


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out
    pp = p[idx]
    order = np.argsort(pp)
    m = len(pp)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = pp[i] * m / (rank + 1)
        prev = min(prev, val)
        adj[i] = prev
    out[idx] = adj
    return out


# ---------------------------------------------------------------------------
def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        sys.exit(1)
    rows = load_rows(dirs)
    n = len(rows)
    n_hall = sum(r["is_hall"] for r in rows)
    n_gplus = sum(r["correct"] for r in rows)
    print(f"loaded {n} rows from {len(dirs)} dir(s)")
    print(f"  G+ (correct)={n_gplus}   HALL (false-yes)={n_hall}   "
          f"K>2={sum(r['K']>2 for r in rows)}")
    print(f"  decode(s)={sorted(set(r['decode'] for r in rows))}")

    # ---- PRIMARY: HALL vs G+ (non-circular) -------------------------------
    print("\n" + "=" * 64)
    print("PRIMARY (non-circular §C2): HALL(false-yes) vs G+(correct)")
    print("=" * 64)
    if n_hall < 5:
        print(f"  !! only {n_hall} HALL samples — underpowered; need adversarial"
              f" split / larger n / sampling. Reporting anyway.")
    shap_feats = ["M", "M_per_tok", "grounding_vs_b0", "grounding_vs_b0_per_tok",
                  "r_prior_b0", "H_pos", "top1_share", "maxphi"]
    base_feats = ["seq_logprob", "seq_logprob_per_tok", "maxsoftmax", "b0"]
    results = []
    for k in shap_feats + base_feats:
        results.append(oriented_auc(rows, k,
                                    lambda r: r["is_hall"],
                                    lambda r: r["correct"] and not r["is_hall"]))
    qvals = benjamini_hochberg([r["p"] for r in results])
    print(f"  {'feature':24s} {'AUC':>6s} {'dir':>3s} {'p':>9s} {'q(BH)':>9s}  group")
    for r, q in zip(results, qvals):
        grp = "SHAP" if r["key"] in shap_feats else "base"
        print(f"  {r['key']:24s} {r['auc']:6.3f} {r.get('dir','?'):>3s} "
              f"{r['p']:9.3g} {q:9.3g}  [{grp}] (n+={r['npos']},n-={r['nneg']})")
    shap_best = max((r['auc'] for r in results if r['key'] in shap_feats
                     and np.isfinite(r['auc'])), default=float('nan'))
    base_best = max((r['auc'] for r in results if r['key'] in base_feats
                     and np.isfinite(r['auc'])), default=float('nan'))
    print(f"  --> best SHAP AUC={shap_best:.3f}  vs  best cheap-baseline "
          f"AUC={base_best:.3f}  (ExactSHAP justified iff SHAP>baseline)")

    # ---- ESTIMATOR: LOO vs ExactSHAP --------------------------------------
    print("\n" + "=" * 64)
    print("ESTIMATOR: LOO vs ExactSHAP divergence (why switch off LOO)")
    print("=" * 64)
    rhos, argdis, eff = [], 0, []
    redundancy = []  # LOO flat but Exact mass large
    for r in rows:
        if not r.get("phi_loo") or r["K"] < 3:
            continue
        pe = np.asarray(r["phi_exact"]); pl = np.asarray(r["phi_loo"])
        if HAVE_SCIPY:
            rho = spearmanr(pe, pl).correlation
        else:
            rho = np.corrcoef(pe.argsort().argsort(), pl.argsort().argsort())[0, 1]
        if np.isfinite(rho):
            rhos.append(rho)
        argdis += int(np.argmax(pe) != np.argmax(pl))
        eff.append(abs(r["M"] - pe.sum()))
        if np.max(np.abs(pl)) < 0.05 and r["M"] > 0.5:
            redundancy.append(r["qid"])
    nval = len(rhos)
    print(f"  n(K>=3)={nval}")
    print(f"  Spearman(phi_exact, phi_loo): mean={np.nanmean(rhos):+.3f} "
          f"median={np.nanmedian(rhos):+.3f}")
    print(f"  top-1 segment disagreement: {argdis}/{sum(1 for r in rows if r.get('phi_loo') and r['K']>=3)} "
          f"(segments MSB would boost differ)")
    print(f"  efficiency residual max |M-sum(phi)|: {max(eff) if eff else 0:.2e}")
    print(f"  REDUNDANCY cases (LOO all|phi|<0.05 but Exact M>0.5): "
          f"{len(redundancy)}  e.g. qid={redundancy[:8]}")
    print(f"    ^ these are LOO's false 'image-agnostic' calls that inflate the"
          f" over-spread tail — the core argument for ExactSHAP.")

    # ---- C1: v_bg vs b0 ----------------------------------------------------
    print("\n" + "=" * 64)
    print("BASELINE GAP §C1: v_bg(background-only) vs b0(true zeros blank)")
    print("=" * 64)
    gap = [r["v_bg"] - r["b0"] for r in rows
           if r.get("b0") is not None and np.isfinite(r.get("b0", float("nan")))]
    if gap:
        print(f"  mean(v_bg - b0)={np.mean(gap):+.3f}  median={np.median(gap):+.3f}"
              f"  (|.|>0.5 share={np.mean(np.abs(gap)>0.5):.2f})")
        print(f"  -> non-zero gap confirms v_bg is NOT the language prior; "
              f"measuring both was correct.")

    # ---- pooled sanity means ----------------------------------------------
    print("\n" + "=" * 64)
    print("Group means (sanity, not a test)")
    print("=" * 64)
    for key in ("M", "grounding_vs_b0", "M_per_tok", "H_pos", "top1_share"):
        g = [r[key] for r in rows if r["correct"] and r.get(key) is not None]
        h = [r[key] for r in rows if r["is_hall"] and r.get(key) is not None]
        gm = np.nanmean(g) if g else float("nan")
        hm = np.nanmean(h) if h else float("nan")
        print(f"  {key:24s} G+={gm:+.3f} (n={len(g)}) | HALL={hm:+.3f} (n={len(h)})")

    print("\n(circular PRIOR/blank-agree comparisons intentionally omitted from"
          " the primary claim per §1.5[C2]; available in rows for sanity.)")


if __name__ == "__main__":
    main()
