r"""Compare LOO (paper) vs ExactSHAP on the diagnostic rows.

Regenerates, side-by-side under BOTH estimators, the two tables the paper's
mass-tail claim rests on:

  Table 4 (mass-tail) : the dominant H-bin and its share of all samples.
  Table 5 (co-location): % of HALL and % of G+ that fall in that dominant bin.

and answers the central re-examination question:

  Does the paper's two-tail story (caption -> over-concentration,
  binary-QA -> over-spread) survive ExactSHAP, or does it collapse?

Paper reference (draft Table 4, LOO):
  CHAIR -> over-concentration (53-66% of samples)
  POPE  -> over-spread        (74-95% of samples)

Also reports the per-sample estimator divergence (why switch off LOO):
  Spearman(phi_loo, phi_exact); top-1 segment disagreement; redundancy cases
  where LOO calls a sample "image-agnostic" (flat phi) but ExactSHAP shows
  large visual mass.

Usage:
  python -m scripts.compare_loo_exact results/exactshap/<cell_dir> [<cell_dir> ...]
  # each <cell_dir> is treated as one (model x bench x decode) cell.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

BIN_NAMES = ["over-conc", "conc", "mixed", "spread", "over-spread"]
PAPER_REF = {  # draft Table 4 LOO dominant tail
    "pope": ("over-spread", "74-95%"),
    "chair": ("over-conc", "53-66%"),
}


def _norm_entropy(phi):
    """Paper's normalized entropy of softmax(phi); mirrors ours_sbc._norm_entropy."""
    import math
    phi = np.asarray(phi, float)
    k = len(phi)
    if k <= 1:
        return 1.0
    z = phi - phi.max()
    p = np.exp(z); p = p / p.sum()
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / math.log(k))


def _bin_of(H):
    if H < 0.2:  return BIN_NAMES[0]
    if H < 0.4:  return BIN_NAMES[1]
    if H < 0.6:  return BIN_NAMES[2]
    if H < 0.8:  return BIN_NAMES[3]
    return BIN_NAMES[4]


def load_rows(d):
    p = Path(d)
    f = p / "rows.jsonl" if p.is_dir() else p
    rows = [json.loads(l) for l in open(f) if l.strip()]
    # Backfill paper-comparable bins for rows dumped by the pre-refactor script
    # (they carry phi_loo / phi_exact but no bin_* fields).
    for r in rows:
        if r.get("bin_exact") is None and r.get("phi_exact") is not None:
            r["bin_exact"] = _bin_of(_norm_entropy(r["phi_exact"]))
        if r.get("bin_loo") is None and r.get("phi_loo") is not None:
            r["bin_loo"] = _bin_of(_norm_entropy(r["phi_loo"]))
    return rows


def tail_table(rows, binkey):
    """dominant bin + share, and HALL/G+ share inside the dominant bin."""
    bins = [r[binkey] for r in rows if r.get(binkey)]
    if not bins:
        return None
    c = Counter(bins)
    dom, dom_n = c.most_common(1)[0]
    n = len(bins)
    halls = [r for r in rows if r.get(binkey) and r["is_hall"]]
    corrs = [r for r in rows if r.get(binkey) and r["correct"]]
    err_in = sum(1 for r in halls if r[binkey] == dom)
    cor_in = sum(1 for r in corrs if r[binkey] == dom)
    return {
        "dom": dom, "dom_share": dom_n / n, "n": n,
        "hist": {b: c.get(b, 0) for b in BIN_NAMES},
        "err_in_dom": (err_in / len(halls)) if halls else float("nan"),
        "cor_in_dom": (cor_in / len(corrs)) if corrs else float("nan"),
        "n_hall": len(halls), "n_corr": len(corrs),
    }


def estimator_divergence(rows):
    rhos, argdis, n3, eff, redun = [], 0, 0, [], []
    for r in rows:
        if not r.get("phi_loo") or r["K"] < 3:
            continue
        n3 += 1
        pe, pl = np.asarray(r["phi_exact"]), np.asarray(r["phi_loo"])
        rho = (spearmanr(pe, pl).correlation if HAVE_SCIPY
               else np.corrcoef(pe.argsort().argsort(), pl.argsort().argsort())[0, 1])
        if np.isfinite(rho):
            rhos.append(rho)
        argdis += int(np.argmax(pe) != np.argmax(pl))
        eff.append(abs(r["M"] - pe.sum()))
        if np.max(np.abs(pl)) < 0.05 and r["M"] > 0.5:
            redun.append(r["qid"])
    return dict(n3=n3, rho_mean=float(np.nanmean(rhos)) if rhos else float("nan"),
                rho_med=float(np.nanmedian(rhos)) if rhos else float("nan"),
                argdis=argdis, eff_max=max(eff) if eff else 0.0, redun=redun)


def fmt_pct(x):
    return f"{x*100:5.1f}%" if x == x else "  n/a"


def main():
    cells = sys.argv[1:]
    if not cells:
        print(__doc__); sys.exit(1)
    for d in cells:
        rows = load_rows(d)
        bench = rows[0].get("bench", "?") if rows else "?"
        name = Path(d).name
        print("\n" + "#" * 72)
        print(f"# {name}   (bench={bench}, n={len(rows)})")
        print("#" * 72)

        loo = tail_table(rows, "bin_loo")
        exa = tail_table(rows, "bin_exact")
        if loo is None or exa is None:
            print("  no binned rows."); continue

        ref = PAPER_REF.get(bench)
        print(f"\n  Paper(LOO) reference dominant tail: "
              f"{ref[0]} ({ref[1]})" if ref else "")
        print(f"\n  {'':14s} {'LOO (paper est.)':>22s}   {'ExactSHAP':>22s}")
        print(f"  {'dominant tail':14s} {loo['dom']:>14s} {fmt_pct(loo['dom_share']):>7s}"
              f"   {exa['dom']:>14s} {fmt_pct(exa['dom_share']):>7s}")
        print(f"  {'HALL in dom':14s} {fmt_pct(loo['err_in_dom']):>22s}"
              f"   {fmt_pct(exa['err_in_dom']):>22s}   (n_HALL={loo['n_hall']})")
        print(f"  {'G+ in dom':14s} {fmt_pct(loo['cor_in_dom']):>22s}"
              f"   {fmt_pct(exa['cor_in_dom']):>22s}   (n_G+={loo['n_corr']})")
        print(f"\n  5-bin histogram [{' '.join(BIN_NAMES)}]")
        print(f"    LOO  : {[loo['hist'][b] for b in BIN_NAMES]}")
        print(f"    Exact: {[exa['hist'][b] for b in BIN_NAMES]}")

        # verdict on the re-examination question
        if loo["dom"] == exa["dom"]:
            verdict = (f"SAME dominant tail ({loo['dom']}); paper's two-tail "
                       f"location is ROBUST to the estimator.")
        else:
            verdict = (f"DIFFERENT: LOO->{loo['dom']} vs Exact->{exa['dom']}. "
                       f"The tail location is an ESTIMATOR ARTIFACT (LOO).")
        print(f"\n  => {verdict}")

        dv = estimator_divergence(rows)
        print(f"\n  estimator divergence (K>=3, n={dv['n3']}): "
              f"Spearman(loo,exact) mean={dv['rho_mean']:+.3f} med={dv['rho_med']:+.3f}")
        print(f"    top-1 segment disagreement: {dv['argdis']}/{dv['n3']}  "
              f"(segments MSB boosts differ)")
        print(f"    efficiency residual max |M-sum(phi)|: {dv['eff_max']:.1e}")
        print(f"    REDUNDANCY (LOO flat |phi|<0.05 but Exact M>0.5): "
              f"{len(dv['redun'])} -> LOO's false 'image-agnostic' calls "
              f"that inflate over-spread; e.g. {dv['redun'][:6]}")

    if len(cells) > 1:
        print("\n" + "=" * 72)
        print("Cross-cell summary: is the two-tail split (caption=conc, "
              "binary-QA=spread) preserved under ExactSHAP?")
        print("=" * 72)
        for d in cells:
            rows = load_rows(d)
            if not rows:
                continue
            loo = tail_table(rows, "bin_loo"); exa = tail_table(rows, "bin_exact")
            print(f"  {Path(d).name:50s} LOO={loo['dom']:>12s}({fmt_pct(loo['dom_share'])})"
                  f"  Exact={exa['dom']:>12s}({fmt_pct(exa['dom_share'])})")


if __name__ == "__main__":
    main()
