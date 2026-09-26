r"""ExactSHAP for VLM hallucination / language-prior diagnosis.

Replaces the paper's leave-one-out (LOO) "pseudo-Shapley" (``ours_msb._shap_phis``)
with the *exact* Shapley value over all ``2^K`` coalitions of the K<=6 panoptic
segments.  Design + review fixes: ``2026-06-08/method_exactshap_fastshap.md``.

Value function (one masking operator for ALL coalitions, no fill drift):

    v(S) = log p( target | image with segments NOT in S occluded by the
                  ORIGINAL image's mean color, filled as a single union )

so v(N) = log p(target | original image)  (grand coalition, nothing hidden)
   v(empty) = v_bg = log p(target | only background visible, all segments masked)

Efficiency axiom holds by construction:  sum_i phi_i = v(N) - v_bg = M.

IMPORTANT (review §1.5[C1]):  v_bg is the "background-only" value, which is NOT
the pure language prior — non-segmented background pixels remain.  The true
prior baselines are measured separately by ``prior_baselines``:
   b0 = log p(target | zeros_like(pixel_values))   # matches PMI/SBC blank
   t0 = log p(target | text only, image tokens removed)  # optional

This module computes attributions only; it forms no claim by itself.  Statistics
and labelling live in ``scripts/diag_exactshap.py``.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..utils.segmentation import Segment, mask_image_with_segments


# ---------------------------------------------------------------------------
# coalition value function
# ---------------------------------------------------------------------------
def _coalition_images(image, segments: Sequence[Segment],
                      subsets: List[frozenset]) -> list:
    """One masked image per coalition S (segments NOT in S occluded, union fill).

    Uses ``mask_image_with_segments`` (single original-image mean, union in one
    shot) so the fill value is identical across coalition sizes — review
    §1.5[M2] / §3.4.
    """
    K = len(segments)
    imgs = []
    for S in subsets:
        hidden = [segments[i].mask for i in range(K) if i not in S]
        imgs.append(mask_image_with_segments(image, hidden))
    return imgs


def _shapley_weight(s: int, K: int) -> float:
    """Shapley coalition weight for |S|=s when attributing to a player not in S:
    s! (K-s-1)! / K!."""
    return math.factorial(s) * math.factorial(K - s - 1) / math.factorial(K)


def shapley_values_from_table(v: Dict[frozenset, float], K: int) -> np.ndarray:
    """Exact Shapley values from a full coalition value table ``v`` (all 2^K
    subsets of range(K) present).  Pure aggregation — no model — so it is unit
    testable against hand-computed cases."""
    phi = np.zeros(K, dtype=np.float64)
    for i in range(K):
        for S, vS in v.items():
            if i in S:
                continue
            phi[i] += _shapley_weight(len(S), K) * (v[S | {i}] - vS)
    return phi


# ---------------------------------------------------------------------------
# exact Shapley
# ---------------------------------------------------------------------------
@torch.no_grad()
def exact_shap_phis(wrapper, image, segments: Sequence[Segment],
                    prompt_ids: torch.Tensor, span: Sequence[int],
                    max_batch: int = 8,
                    with_interaction: bool = False
                    ) -> Optional[Dict]:
    """Exact Shapley values over the 2^K segment coalitions.

    Returns ``None`` when the game is degenerate (K<2 segments or empty span),
    matching the greedy fallback in ``ours_sbc`` (review §1.5[Gaps]).  Otherwise:

        {
          'phi':   np.ndarray (K,)   exact Shapley value per segment,
          'v_bg':  float             v(empty) = background-only log p,
          'v_N':   float             v(N)     = original-image log p,
          'M':     float             v_N - v_bg = sum(phi)  (total visual mass),
          'v':     {frozenset: float}  full value table (for downstream / debug),
          'interaction': np.ndarray (K,K)  (only if with_interaction)
        }

    Cost: 2^K teacher-forced forwards delegated to ``wrapper.logp_spans``
    (LLaVA batches on the pixel axis; Qwen loops per image — see review §1.5
    [비용 정정]).
    """
    K = len(segments)
    span = list(span)
    if K < 2 or not span:
        return None

    subsets = [frozenset(c) for r in range(K + 1) for c in combinations(range(K), r)]
    imgs = _coalition_images(image, segments, subsets)
    vals = wrapper.logp_spans(imgs, prompt_ids, span, max_batch=max_batch)
    vals = [float(x) for x in vals]
    v: Dict[frozenset, float] = {S: val for S, val in zip(subsets, vals)}

    phi = shapley_values_from_table(v, K)

    v_bg = v[frozenset()]
    v_N = v[frozenset(range(K))]
    M = v_N - v_bg
    # efficiency axiom (exact up to fp accumulation)
    assert abs(M - phi.sum()) < 1e-3, f"efficiency violated: {M} vs {phi.sum()}"

    out = dict(phi=phi, v_bg=v_bg, v_N=v_N, M=M, v=v)
    if with_interaction:
        out["interaction"] = _shapley_interaction(v, K)
    return out


@torch.no_grad()
def coalition_pertoken_logp(wrapper, images, prompt_ids, caption_ids,
                            max_batch: int = 8) -> np.ndarray:
    """Per-token teacher-forced log p for EACH coalition image (LLaVA path).

    Returns an array of shape ``(len(images), L)`` where ``L=len(caption_ids)``
    and entry [i, t] = log p(caption_ids[t] | prompt + caption[:t], image_i).

    This is the per-position generalization of ``LlavaWrapper.logp_spans`` (which
    sums over positions): one forward over (prompt + full caption) per coalition
    image yields the log-prob at EVERY caption position, so v_c(S) for *all*
    objects in the caption is extracted from the SAME 2^K forwards (object
    localization happens in the caller).  LLaVA-only (fixed 576 visual tokens,
    no image_grid_thw); the targeted v_c pilot runs on LLaVA-1.5-7B.
    """
    device, dtype = wrapper.device, wrapper.dtype
    L = len(caption_ids)
    pix_list = []
    for im in images:
        enc = wrapper.processor.image_processor(images=im, return_tensors="pt")
        pix_list.append(enc["pixel_values"].to(device, dtype=dtype))
    pixel_batch = torch.cat(pix_list, dim=0)
    span_t = torch.tensor([list(caption_ids)], device=prompt_ids.device,
                          dtype=prompt_ids.dtype)
    full_ids = torch.cat([prompt_ids, span_t], dim=1)
    cap_idx = torch.tensor(list(caption_ids), device=device, dtype=torch.long)
    attn_ones = torch.ones_like(full_ids)
    out_rows = []
    N = pixel_batch.shape[0]
    for start in range(0, N, max_batch):
        chunk = pixel_batch[start:start + min(max_batch, N - start)]
        b = chunk.shape[0]
        out = wrapper.model(input_ids=full_ids.repeat(b, 1), pixel_values=chunk,
                            attention_mask=attn_ones.repeat(b, 1),
                            use_cache=False, return_dict=True)
        logits = out.logits.float()                       # (b, S, V)
        span_logits = logits[:, -L - 1:-1, :]             # (b, L, V)
        lp = torch.log_softmax(span_logits, dim=-1)
        per_tok = lp[:, torch.arange(L, device=device), cap_idx]   # (b, L)
        out_rows.append(per_tok.cpu())
        del out, logits, lp
    return torch.cat(out_rows, dim=0).numpy()             # (N, L)


def exact_shap_from_coalition_values(coalition_vals, subsets, K):
    """Exact Shapley from a list of v(S) aligned to ``subsets`` order.
    Returns (phi, v_bg, v_N, M)."""
    v = {S: float(val) for S, val in zip(subsets, coalition_vals)}
    phi = shapley_values_from_table(v, K)
    v_bg, v_N = v[frozenset()], v[frozenset(range(K))]
    return phi, v_bg, v_N, (v_N - v_bg)


def all_subsets(K):
    return [frozenset(c) for r in range(K + 1) for c in combinations(range(K), r)]


def loo_from_coalition_values(coalition_vals, subsets, K):
    """Leave-one-out terms phi_i = v(N) - v(N\\{i}) extracted from the SAME
    coalition value list used by ExactSHAP (free — no extra forwards).  Lets the
    object-level v_c diagnostic report LOO and ExactSHAP side by side."""
    v = {S: float(val) for S, val in zip(subsets, coalition_vals)}
    full = frozenset(range(K))
    vN = v[full]
    return np.array([vN - v[full - {i}] for i in range(K)], dtype=np.float64)


def _shapley_interaction(v: Dict[frozenset, float], K: int) -> np.ndarray:
    """Shapley interaction index I_{ij} (Grabisch-Roubens), symmetric KxK,
    diagonal = 0.  Weight w(s) = s!(K-s-2)!/(K-1)! over S subset of N\\{i,j}.
    Captures redundancy (I<0) / synergy (I>0) that LOO cannot see."""
    I = np.zeros((K, K), dtype=np.float64)
    others = list(range(K))
    for i in range(K):
        for j in range(i + 1, K):
            rest = [x for x in others if x != i and x != j]
            acc = 0.0
            for r in range(len(rest) + 1):
                for c in combinations(rest, r):
                    S = frozenset(c)
                    w = (math.factorial(r) * math.factorial(K - r - 2)
                         / math.factorial(K - 1))
                    acc += w * (v[S | {i, j}] - v[S | {i}] - v[S | {j}] + v[S])
            I[i, j] = I[j, i] = acc
    return I


# ---------------------------------------------------------------------------
# LOO (paper baseline) under the SAME masking operator, for a clean comparison
# ---------------------------------------------------------------------------
@torch.no_grad()
def loo_phis(wrapper, image, segments: Sequence[Segment],
             prompt_ids: torch.Tensor, span: Sequence[int],
             max_batch: int = 8) -> Optional[np.ndarray]:
    """Leave-one-out term phi_i = v(N) - v(N\\{i}), using the union-fill operator
    so LOO and ExactSHAP differ only in the estimator, not the masking (review
    §1.5[M2])."""
    K = len(segments)
    span = list(span)
    if K < 2 or not span:
        return None
    full = frozenset(range(K))
    subsets = [full] + [full - {i} for i in range(K)]
    imgs = _coalition_images(image, segments, subsets)
    vals = [float(x) for x in wrapper.logp_spans(imgs, prompt_ids, span,
                                                 max_batch=max_batch)]
    v_N = vals[0]
    return np.asarray([v_N - vals[1 + i] for i in range(K)], dtype=np.float64)


# ---------------------------------------------------------------------------
# prior baselines (NOT part of the Shapley game; review §1.5[C1])
# ---------------------------------------------------------------------------
def _extra_from_enc(enc) -> dict:
    """Model-specific forward extras (Qwen: image_grid_thw, mm_token_type_ids).
    LLaVA has none."""
    extra = {}
    for k in ("image_grid_thw", "mm_token_type_ids"):
        if k in enc and enc[k] is not None:
            extra[k] = enc[k]
    return extra


@torch.no_grad()
def prior_baselines(wrapper, enc, span: Sequence[int],
                    text_only: bool = False) -> Dict[str, Optional[float]]:
    """Prior-strength baselines, measured outside the Shapley game.

    b0 = log p(span | zeros_like(pixel_values))  -- the EXACT blank used by
         PMI/SBC (``torch.zeros_like(pixel_v)``), i.e. featureless mid-gray in
         CLIP-normalized space.  This is the prior reference the paper already
         ships, so SHAP's grounding can be compared against it apples-to-apples.

    t0 = log p(span | x, no image)  -- text-only LM baseline (image tokens
         dropped).  Model-specific; only attempted when ``text_only=True`` and
         the wrapper exposes ``logp_span_text_only`` (else returned as None).
    """
    from .ours_v3 import _logp_span_under_image

    span = list(span)
    if not span:
        return {"b0": 0.0, "t0": 0.0 if text_only else None}

    input_ids = enc["input_ids"]
    pixel_v = enc["pixel_values"]
    attn = enc.get("attention_mask")
    extra = _extra_from_enc(enc)

    b0 = _logp_span_under_image(wrapper, input_ids, torch.zeros_like(pixel_v),
                                attn, span, **extra)

    t0 = None
    if text_only and hasattr(wrapper, "logp_span_text_only"):
        t0 = float(wrapper.logp_span_text_only(input_ids, span))
    return {"b0": float(b0), "t0": t0}


# ---------------------------------------------------------------------------
# scalar summary features (review §4 + §1.5[M4] length normalization)
# ---------------------------------------------------------------------------
def signature_features(res: Dict, b0: Optional[float] = None,
                       span_len: int = 1) -> Dict[str, float]:
    """Derive the label-free detector features from an ``exact_shap_phis`` result.

    All log-prob magnitudes are reported both raw and per-token (``/span_len``)
    so the within-input M comparison is not confounded by span length
    (review §1.5[M4]).  ``r_prior`` is length-invariant in expectation.
    """
    phi = res["phi"]
    v_bg, v_N, M = res["v_bg"], res["v_N"], res["M"]
    L = max(int(span_len), 1)

    pos = np.clip(phi, 0.0, None)
    pos_sum = float(pos.sum())
    if pos_sum > 0 and len(phi) > 1:
        p = pos / pos_sum
        p = np.clip(p, 1e-12, 1.0)
        H_pos = float(-(p * np.log(p)).sum() / math.log(len(phi)))
        gini = float(np.sort(p)[::-1][0])          # top-1 share
    else:
        H_pos, gini = 1.0, 0.0

    feats = {
        "M": M,
        "M_per_tok": M / L,
        "v_bg": v_bg,
        "v_N": v_N,
        "maxphi": float(phi.max()),
        "minphi": float(phi.min()),               # signed: distractor effect
        "H_pos": H_pos,
        "top1_share": gini,
        "K": float(len(phi)),
    }
    # prior reference vs the SAME-game background and vs the true blank b0
    feats["r_prior_bg"] = math.exp(min(v_bg - v_N, 0.0))   # bg explains output
    if b0 is not None:
        feats["b0"] = b0
        feats["grounding_vs_b0"] = v_N - b0               # image over true prior
        feats["grounding_vs_b0_per_tok"] = (v_N - b0) / L
        feats["r_prior_b0"] = math.exp(min(b0 - v_N, 0.0))
    return feats
