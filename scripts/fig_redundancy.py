r"""Main figure: why LOO collapses under redundancy (theory) + real-data confirm."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def shapley_loo(K, r):
    ev = set(range(r))
    def v(S): return 1.0 if (set(S) & ev) else 0.0
    full = frozenset(range(K))
    M = v(full) - v(frozenset())
    loo = sum(v(full) - v(full - {i}) for i in range(K))
    return M, loo


def auc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def load_first(dirs):
    rows = []
    for d in dirs:
        p = Path("results/exactshap") / d / "rows.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in open(p) if l.strip()]
    byimg = defaultdict(list)
    for r in rows: byimg[r["iid"]].append(r)
    first = []
    for iid, os_ in byimg.items():
        seen = set()
        for o in os_:
            if o["obj"] not in seen: seen.add(o["obj"]); first.append(o)
    return [r for r in first if not r["hallucinated"] and "M_c_loo" in r]


fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel A: theory — AUC vs redundant fraction
K = 6; rng = np.random.default_rng(0)
fracs = np.linspace(0, 1, 11); ae, al = [], []
for f in fracs:
    gE, gL = [], []
    for _ in range(400):
        r = 2 + rng.integers(0, K - 1) if rng.random() < f else 1
        M, loo = shapley_loo(K, r); gE.append(M); gL.append(loo)
    hE = [0.0] * 400; hL = [0.0] * 400
    ae.append(auc(gE, hE)); al.append(auc(gL, hL))
ax[0].plot(fracs, ae, "o-", label="ExactSHAP (with v(∅))", lw=2)
ax[0].plot(fracs, al, "s--", label="LOO", lw=2, color="crimson")
ax[0].axhline(0.5, color="gray", ls=":", lw=1)
ax[0].set_xlabel("fraction of grounded objects with redundant evidence (r≥2)")
ax[0].set_ylabel("AUC (grounded vs hallucinated)")
ax[0].set_title("(A) Theory: LOO collapses under redundancy")
ax[0].set_ylim(0.45, 1.03); ax[0].legend(loc="lower left"); ax[0].grid(alpha=.3)

# Panel B: real data — LOO under-estimation gap vs redundancy quartile
for name, dirs, col in [
    ("CHAIR", ["llava-1.5-7b-hf_chair_vc_greedy_seed1234",
               "llava-1.5-7b-hf_chair_vc_greedy_seed1234_off300"], "tab:blue"),
    ("AMBER", ["llava-1.5-7b-hf_amber_vc_greedy_seed1234",
               "llava-1.5-7b-hf_amber_vc_greedy_seed1234_off502"], "tab:green")]:
    rows = load_first(dirs)
    spread = np.array([r["M_c"] - r["maxphi_c"] for r in rows])
    gap = np.array([r["M_c"] - r["M_c_loo"] for r in rows])
    q = np.quantile(spread, [0, .25, .5, .75, 1.0])
    means = []
    for i in range(4):
        m = (spread >= q[i]) & ((spread <= q[i+1]) if i == 3 else (spread < q[i+1]))
        means.append(gap[m].mean())
    ax[1].plot(range(1, 5), means, "o-", label=f"{name} (ρ≈{np.corrcoef(spread.argsort().argsort(), gap.argsort().argsort())[0,1]:.2f})", color=col, lw=2)
ax[1].axhline(0, color="gray", ls=":", lw=1)
ax[1].set_xticks(range(1, 5)); ax[1].set_xlabel("evidence-redundancy quartile (low→high)")
ax[1].set_ylabel("LOO under-estimation gap  M_c − Σφ_LOO")
ax[1].set_title("(B) Real data: gap grows with redundancy")
ax[1].legend(loc="upper left"); ax[1].grid(alpha=.3)

plt.tight_layout()
out = "2026-06-08/fig_redundancy.png"
plt.savefig(out, dpi=150)
print("wrote", out)
