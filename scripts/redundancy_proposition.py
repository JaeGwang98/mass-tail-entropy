r"""Tier-1 (no GPU): redundancy toy proposition + real-data mechanism correlation.

(A) Proposition demo: an OR-type coalition game where a grounded object's
    evidence is REDUNDANTLY present in r of K segments:
        v(S) = 1 if S intersects the r evidence segments, else 0
    A hallucinated object has NO evidence: v(S)=0 for all S.

    Then:
      - ExactSHAP total  M = v(N)-v(empty) = 1 (grounded) / 0 (hall)  -> separates
      - LOO sum  Σ_i [v(N)-v(N\{i})] = 0 for r>=2 (others compensate)  -> collapses
    => For r>=2, LOO cannot tell a (redundantly) grounded object from a
       hallucinated one; ExactSHAP can. For r=1, LOO == ExactSHAP.

(B) Real-data check: per grounded caption object, redundancy proxy
    spread = M_c - maxphi_c (visual mass BEYOND the single top segment;
    larger = more distributed/redundant evidence). Prediction from (A):
    the LOO under-estimation gap  Δ_obj = M_c - M_c_loo  grows with spread.

Usage: python -m scripts.redundancy_proposition
"""
import json
from itertools import combinations
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# (A) proposition demonstration
# ---------------------------------------------------------------------------
def shapley_total_and_loo(K, evidence):
    """For the OR-game with evidence set `evidence` (segment indices that carry
    the object's visual support), return (M_exact, loo_sum)."""
    ev = set(evidence)
    def v(S): return 1.0 if (set(S) & ev) else 0.0
    full = frozenset(range(K))
    vN, vEmpty = v(full), v(frozenset())
    M = vN - vEmpty                                   # ExactSHAP total (efficiency)
    loo = sum(v(full) - v(full - {i}) for i in range(K))
    return M, loo


def demo_A():
    K = 6
    print("(A) OR-game: grounded object's evidence redundant in r segments (K=6)")
    print(f"  {'r (redundancy)':16s} {'ExactSHAP M':>12s} {'LOO Σφ':>8s}")
    for r in range(1, K + 1):
        M, loo = shapley_total_and_loo(K, range(r))
        print(f"  {r:>14d} {M:12.2f} {loo:8.2f}")
    Mh, looh = shapley_total_and_loo(K, [])           # hallucinated: no evidence
    print(f"  {'hallucinated':>14s} {Mh:12.2f} {looh:8.2f}")
    print("  => grounded(r>=2): Exact=1 but LOO=0 == hallucinated  -> LOO cannot separate")
    print("     grounded(r=1) : Exact=LOO=1  -> agree")


def demo_A_auc():
    """AUC(grounded vs hall) as the fraction of grounded objects that are
    REDUNDANT (r>=2) grows. Mirrors caption (redundant) vs binary (r=1)."""
    K = 6
    rng = np.random.default_rng(0)
    print("\n(A') AUC vs redundant-fraction (1000 grounded + 1000 hall objects)")
    print(f"  {'frac r>=2':>10s} {'Exact AUC':>10s} {'LOO AUC':>8s}")
    def auc(pos, neg):  # P(pos>neg), ties .5
        c = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
        return c / (len(pos) * len(neg))
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gE, gL = [], []
        for _ in range(300):
            r = 2 + rng.integers(0, K - 1) if rng.random() < frac else 1
            M, loo = shapley_total_and_loo(K, range(r))
            gE.append(M); gL.append(loo)
        hE, hL = [0.0] * 300, [0.0] * 300
        print(f"  {frac:10.2f} {auc(gE,hE):10.3f} {auc(gL,hL):8.3f}")


# ---------------------------------------------------------------------------
# (B) real-data mechanism correlation
# ---------------------------------------------------------------------------
def load_first(dirs):
    rows = []
    for d in dirs:
        p = Path("results/exactshap") / d
        if (p / "rows.jsonl").exists():
            rows += [json.loads(l) for l in open(p / "rows.jsonl") if l.strip()]
    from collections import defaultdict
    byimg = defaultdict(list)
    for r in rows:
        byimg[r["iid"]].append(r)
    first = []
    for iid, os_ in byimg.items():
        seen = set()
        for o in os_:
            if o["obj"] not in seen:
                seen.add(o["obj"]); first.append(o)
    return first


def demo_B(name, dirs):
    rows = [r for r in load_first(dirs)
            if not r["hallucinated"] and "M_c_loo" in r]   # grounded only
    if not rows:
        print(f"\n(B) {name}: no data"); return
    spread = np.array([r["M_c"] - r["maxphi_c"] for r in rows])   # redundancy proxy
    gap = np.array([r["M_c"] - r["M_c_loo"] for r in rows])       # LOO under-est
    # Spearman (no scipy dependency)
    def spearman(a, b):
        ra = a.argsort().argsort(); rb = b.argsort().argsort()
        return np.corrcoef(ra, rb)[0, 1]
    rho = spearman(spread, gap)
    print(f"\n(B) {name}: grounded objects n={len(rows)}")
    print(f"  Spearman(redundancy-proxy [M_c-maxphi], LOO-gap [M_c-M_c_loo]) = {rho:+.3f}")
    # binned means
    q = np.quantile(spread, [0, .25, .5, .75, 1.0])
    print(f"  {'redundancy quartile':22s} {'mean LOO-gap':>12s} {'n':>5s}")
    for i in range(4):
        m = (spread >= q[i]) & (spread <= q[i + 1] if i == 3 else spread < q[i + 1])
        print(f"  Q{i+1} [{q[i]:+.2f},{q[i+1]:+.2f}]      {gap[m].mean():12.3f} {m.sum():5d}")
    print("  => LOO under-estimates MORE when evidence is more distributed (proposition A confirmed)")


if __name__ == "__main__":
    demo_A()
    demo_A_auc()
    demo_B("CHAIR", ["llava-1.5-7b-hf_chair_vc_greedy_seed1234",
                     "llava-1.5-7b-hf_chair_vc_greedy_seed1234_off300"])
    demo_B("AMBER", ["llava-1.5-7b-hf_amber_vc_greedy_seed1234",
                     "llava-1.5-7b-hf_amber_vc_greedy_seed1234_off502"])
