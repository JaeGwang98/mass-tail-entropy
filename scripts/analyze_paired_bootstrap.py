r"""Tier-1 significance analysis (no GPU): paired bootstrap on AUC_exact - AUC_loo
for the caption v_c object-level test, + DeLong test for single AUC vs 0.5.

LOO and ExactSHAP score the SAME objects, so the two AUCs are PAIRED — we must
bootstrap the DIFFERENCE, not test them independently.  Uses only stored M_c /
M_c_loo (no model inference).

Usage:
  python -m scripts.analyze_paired_bootstrap
"""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

B = 10000
SEED = 0


def load_first_mention(dirs, exact_key, loo_key):
    rows = []
    for d in dirs:
        p = Path("results/exactshap") / d
        if (p / "rows.jsonl").exists():
            rows += [json.loads(l) for l in open(p / "rows.jsonl") if l.strip()]
    byimg = defaultdict(list)
    for r in rows:
        byimg[r["iid"]].append(r)
    first = []
    for iid, os_ in byimg.items():
        seen = set()
        for o in os_:
            if o["obj"] not in seen:
                seen.add(o["obj"]); first.append(o)
    H = [r for r in first if r["hallucinated"]]
    G = [r for r in first if not r["hallucinated"]]
    # hallucination = LOW visual support, so "positive score for hall" = -value
    # We orient AUC as P(grounded ranks higher than hall) on the raw value, then
    # report max(auc,1-auc) consistently. Keep raw arrays per method.
    def arr(rs, k): return np.array([r[k] for r in rs], float)
    return (arr(H, exact_key), arr(G, exact_key),
            arr(H, loo_key), arr(G, loo_key))


def auc(pos, neg):
    """P(pos > neg) with tie=0.5. Here 'pos'=grounded, 'neg'=hall (grounded
    should score higher under the under-grounding hypothesis)."""
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    s = allv[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r1 = ranks[:n1].sum()
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def delong_var_single(pos, neg):
    """Variance of a single AUC (DeLong). pos=grounded, neg=hall."""
    m, n = len(pos), len(neg)
    # structural components
    def psi(a, b):  # 1 if a>b, .5 if =, 0 if <  -> via broadcasting
        return (a[:, None] > b[None, :]) * 1.0 + (a[:, None] == b[None, :]) * 0.5
    P = psi(pos, neg)               # (m,n)
    theta = P.mean()
    v10 = P.mean(axis=1)            # per positive
    v01 = P.mean(axis=0)            # per negative
    s10 = v10.var(ddof=1) if m > 1 else 0.0
    s01 = v01.var(ddof=1) if n > 1 else 0.0
    var = s10 / m + s01 / n
    return theta, var


def delong_p_vs_half(pos, neg):
    from math import erf, sqrt
    theta, var = delong_var_single(pos, neg)
    if var <= 0:
        return theta, float("nan")
    z = (theta - 0.5) / sqrt(var)
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return theta, p


def paired_bootstrap(He, Ge, Hl, Gl):
    """Resample objects (stratified by class), same indices for both methods.
    He/Hl = hall scores (exact/loo), Ge/Gl = grounded scores. AUC oriented as
    P(grounded > hall) so >0.5 means grounded scores higher (under-grounding)."""
    rng = np.random.default_rng(SEED)
    nH, nG = len(He), len(Ge)
    ae0, al0 = auc(Ge, He), auc(Gl, Hl)
    deltas = np.empty(B); aes = np.empty(B); als = np.empty(B)
    for b in range(B):
        ih = rng.integers(0, nH, nH)
        ig = rng.integers(0, nG, nG)
        ae = auc(Ge[ig], He[ih]); al = auc(Gl[ig], Hl[ih])
        aes[b], als[b], deltas[b] = ae, al, ae - al
    def ci(x): return np.percentile(x, [2.5, 97.5])
    return dict(auc_exact=ae0, auc_loo=al0,
                ci_exact=ci(aes), ci_loo=ci(als),
                delta=ae0 - al0, ci_delta=ci(deltas),
                p_delta_gt0=float((deltas > 0).mean()))


def report(name, dirs, exact_key="M_c", loo_key="M_c_loo"):
    He, Ge, Hl, Gl = load_first_mention(dirs, exact_key, loo_key)
    print(f"\n### {name}  (hall={len(He)}, grounded={len(Ge)})  feature={exact_key}")
    r = paired_bootstrap(He, Ge, Hl, Gl)
    # DeLong vs 0.5 for each method (orient grounded>hall)
    _, p_ex = delong_p_vs_half(Ge, He)
    _, p_lo = delong_p_vs_half(Gl, Hl)
    print(f"  ExactSHAP AUC = {r['auc_exact']:.3f}  95%CI[{r['ci_exact'][0]:.3f},{r['ci_exact'][1]:.3f}]  DeLong p(vs0.5)={p_ex:.2e}")
    print(f"  LOO       AUC = {r['auc_loo']:.3f}  95%CI[{r['ci_loo'][0]:.3f},{r['ci_loo'][1]:.3f}]  DeLong p(vs0.5)={p_lo:.2e}")
    print(f"  PAIRED Δ(Exact-LOO) = {r['delta']:.3f}  95%CI[{r['ci_delta'][0]:.3f},{r['ci_delta'][1]:.3f}]  P(Δ>0)={r['p_delta_gt0']:.4f}")
    sig = "✅ Δ CI excludes 0" if r['ci_delta'][0] > 0 else "⚠️ Δ CI includes 0"
    print(f"  -> {sig}")
    return r


if __name__ == "__main__":
    print(f"Paired bootstrap (B={B}) on AUC(Exact)-AUC(LOO), caption v_c first-mention")
    print("AUC oriented as P(grounded > hall); >0.5 = grounded has MORE visual support")
    report("CHAIR (LLaVA)",
           ["llava-1.5-7b-hf_chair_vc_greedy_seed1234",
            "llava-1.5-7b-hf_chair_vc_greedy_seed1234_off300"])
    report("AMBER (LLaVA)",
           ["llava-1.5-7b-hf_amber_vc_greedy_seed1234",
            "llava-1.5-7b-hf_amber_vc_greedy_seed1234_off502"])
    # also maxphi feature
    report("CHAIR (LLaVA) [maxphi]",
           ["llava-1.5-7b-hf_chair_vc_greedy_seed1234",
            "llava-1.5-7b-hf_chair_vc_greedy_seed1234_off300"],
           exact_key="maxphi_c", loo_key="maxphi_c_loo")
    report("AMBER (LLaVA) [maxphi]",
           ["llava-1.5-7b-hf_amber_vc_greedy_seed1234",
            "llava-1.5-7b-hf_amber_vc_greedy_seed1234_off502"],
           exact_key="maxphi_c", loo_key="maxphi_c_loo")
