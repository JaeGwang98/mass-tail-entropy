r"""CPU-only unit tests for the ExactSHAP math — no VLM, no GPU.

Verifies, against hand-computed ground truth:
  1. Shapley weights and the efficiency axiom (sum phi == v(N) - v(empty)).
  2. K=2 closed form: phi_i = 1/2[(v(i)-v(empty)) + (v(N)-v(j))].
  3. K=3 additive game: Shapley == marginal (interaction == 0).
  4. Shapley interaction index: symmetry, zero on additive games, and the
     redundancy sign (I<0) on a saturating (OR-like) game.
  5. union mean-fill is order-independent and unbiased (review §1.5[M2]).
  6. exact_shap_phis plumbing + None guards via a mock wrapper.

Run:  python -m scripts.test_exact_shap_math   (from repo root)
  or: python scripts/test_exact_shap_math.py
"""
import math
import os
import sys
from itertools import combinations

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.decoding.exact_shap import (          # noqa: E402
    shapley_values_from_table, _shapley_interaction, _shapley_weight,
    exact_shap_phis, signature_features,
)
from src.utils.segmentation import (           # noqa: E402
    Segment, mask_image_with_segment, mask_image_with_segments,
)

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def full_table(K, fn):
    """Build {frozenset(S): fn(S)} over all subsets of range(K)."""
    v = {}
    for r in range(K + 1):
        for c in combinations(range(K), r):
            v[frozenset(c)] = float(fn(frozenset(c)))
    return v


# --------------------------------------------------------------------------
print("1. weights + efficiency axiom")
# random-ish but deterministic value function over K=4
rng = np.random.default_rng(0)
K = 4
vals = {frozenset(c): float(rng.standard_normal())
        for r in range(K + 1) for c in combinations(range(K), r)}
phi = shapley_values_from_table(vals, K)
M = vals[frozenset(range(K))] - vals[frozenset()]
check("efficiency: sum(phi) == v(N)-v(empty)", abs(phi.sum() - M) < 1e-9,
      f"{phi.sum()} vs {M}")
# weight sanity: sum over s of C(K-1,s)*weight(s) == 1 (a player's weights sum to 1)
wsum = sum(math.comb(K - 1, s) * _shapley_weight(s, K) for s in range(K))
check("player weights sum to 1", abs(wsum - 1.0) < 1e-12, str(wsum))


# --------------------------------------------------------------------------
print("2. K=2 closed form")
v2 = {frozenset(): 0.3, frozenset({0}): 1.0, frozenset({1}): 0.5,
      frozenset({0, 1}): 2.0}
phi2 = shapley_values_from_table(v2, 2)
exp0 = 0.5 * ((v2[frozenset({0})] - v2[frozenset()]) +
              (v2[frozenset({0, 1})] - v2[frozenset({1})]))
exp1 = 0.5 * ((v2[frozenset({1})] - v2[frozenset()]) +
              (v2[frozenset({0, 1})] - v2[frozenset({0})]))
check("phi_0 closed form", abs(phi2[0] - exp0) < 1e-12, f"{phi2[0]} vs {exp0}")
check("phi_1 closed form", abs(phi2[1] - exp1) < 1e-12, f"{phi2[1]} vs {exp1}")
check("K=2 efficiency", abs(phi2.sum() - (2.0 - 0.3)) < 1e-12)


# --------------------------------------------------------------------------
print("3. additive game -> Shapley == singleton marginals, interaction == 0")
a = np.array([0.7, -0.4, 1.1])           # per-player additive contributions
K = 3
v_add = full_table(K, lambda S: sum(a[i] for i in S))
phi_add = shapley_values_from_table(v_add, K)
check("additive: phi == a", np.allclose(phi_add, a, atol=1e-12),
      f"{phi_add} vs {a}")
I_add = _shapley_interaction(v_add, K)
check("additive: interaction == 0", np.allclose(I_add, 0.0, atol=1e-12))
check("additive: efficiency", abs(phi_add.sum() - a.sum()) < 1e-12)


# --------------------------------------------------------------------------
print("4. interaction: symmetry + redundancy sign on OR/saturating game")
# v(S) = 1 if S non-empty else 0  (max saturating -> redundant players)
K = 3
v_or = full_table(K, lambda S: 1.0 if len(S) > 0 else 0.0)
I_or = _shapley_interaction(v_or, K)
check("interaction symmetric", np.allclose(I_or, I_or.T, atol=1e-12))
check("interaction diag zero", np.allclose(np.diag(I_or), 0.0))
# redundant players -> negative pairwise interaction
offdiag = I_or[np.triu_indices(K, k=1)]
check("redundant game: interaction < 0", np.all(offdiag < 0),
      str(offdiag))
phi_or = shapley_values_from_table(v_or, K)
check("OR game efficiency", abs(phi_or.sum() - 1.0) < 1e-12)
check("OR game symmetric phi", np.allclose(phi_or, phi_or[0]))


# --------------------------------------------------------------------------
print("5. union mean-fill: order-independent + unbiased vs single original mean")
img = Image.fromarray((rng.integers(0, 256, (16, 16, 3))).astype(np.uint8))
m0 = np.zeros((16, 16), bool); m0[:8, :8] = True
m1 = np.zeros((16, 16), bool); m1[8:, 8:] = True      # disjoint from m0
u_ab = np.asarray(mask_image_with_segments(img, [m0, m1]))
u_ba = np.asarray(mask_image_with_segments(img, [m1, m0]))
check("union order-independent", np.array_equal(u_ab, u_ba))
# the fill value equals the ORIGINAL image mean (not a drifted mean)
orig_mean = np.asarray(img).reshape(-1, 3).mean(axis=0).astype(np.uint8)
filled_px = u_ab[m0 | m1]
check("union fill == original mean",
      np.all(filled_px == orig_mean), f"{filled_px[0]} vs {orig_mean}")
# sequential per-call masking DRIFTS (this is the bias §1.5[M2] removes)
seq = mask_image_with_segment(mask_image_with_segment(img, m0), m1)
seq_fill = np.asarray(seq)[m1]
check("sequential masking drifts (bias exists)",
      not np.all(seq_fill == orig_mean),
      "expected drift; if equal, masks may overlap")
check("empty union returns original", np.array_equal(
      np.asarray(mask_image_with_segments(img, [])), np.asarray(img)))


# --------------------------------------------------------------------------
print("6. exact_shap_phis plumbing + None guards (mock wrapper)")


class MockWrapper:
    """logp_spans returns a deterministic value per coalition image, keyed by
    how many segment-pixels remain visible (monotone, additive-ish)."""
    def __init__(self, base):
        self.base = base  # original image array sum as reference

    def logp_spans(self, images, prompt_ids, span, max_batch=8):
        out = []
        for im in images:
            arr = np.asarray(im).astype(np.float64)
            # value = -(distance of this coalition image from original)/scale
            out.append(-np.abs(arr - self.base).sum() / 1e5)
        return np.asarray(out)


base_arr = np.asarray(img).astype(np.float64)
segs = [Segment(mask=m0, area_frac=0.25), Segment(mask=m1, area_frac=0.25)]
mock = MockWrapper(base_arr)
res = exact_shap_phis(mock, img, segs, prompt_ids=None, span=[1, 2, 3])
check("returns dict for K>=2", isinstance(res, dict))
check("efficiency in exact_shap_phis",
      abs(res["M"] - res["phi"].sum()) < 1e-6, f"{res['M']} vs {res['phi'].sum()}")
check("v_N is grand coalition (==0 distance for original)",
      abs(res["v_N"] - 0.0) < 1e-9, str(res["v_N"]))
check("v_bg <= v_N (occluding can't help here)", res["v_bg"] <= res["v_N"] + 1e-9)
check("None guard: K<2", exact_shap_phis(mock, img, segs[:1], None, [1]) is None)
check("None guard: empty span", exact_shap_phis(mock, img, segs, None, []) is None)

feats = signature_features(res, b0=-1.23, span_len=3)
check("features: M_per_tok == M/L", abs(feats["M_per_tok"] - res["M"] / 3) < 1e-12)
check("features: grounding_vs_b0 == v_N-b0",
      abs(feats["grounding_vs_b0"] - (res["v_N"] - (-1.23))) < 1e-12)
check("features: 0<=H_pos<=1", 0.0 <= feats["H_pos"] <= 1.0)


# --------------------------------------------------------------------------
print(f"\n{'='*50}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
