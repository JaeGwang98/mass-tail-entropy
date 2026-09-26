"""OPERA-Greedy — Over-Trust Penalty (OTP) variant of OPERA without the
beam-search retrospection.

Mechanism (paper Eq. 3–6, Huang et al. CVPR 2024):
  1. Cut a window of the last ``k_window`` *decoded text* tokens (image and
     prompt tokens are excluded — see paper §3.2).
  2. Take the last-layer self-attention, take the per-head max, and scale by
     ``sigma`` to bring small values into a useful range.
  3. For each column c in the lower triangle, compute the product of the
     scaled attention values down that column; ``phi`` is the largest such
     column product.
  4. Use ``phi`` to discriminate among the top-N candidates of the *current*
     step. Because the OPERA penalty is a scalar of the per-beam history,
     in a single-path (greedy) decoder it would cancel from the argmax — so
     we evaluate each candidate by doing one forward step *with that
     candidate*, recompute ``phi`` for the extended sequence, and pick the
     candidate that maximizes ``logit(c) − α·log(1+phi_c)``.

Greedy adaptation note: the raw column-product of OPERA can range over 6+
orders of magnitude (1e-3 to 1e5+). In beam search this is benign because
the penalty competes with *cumulative* log-probabilities, but in a single-
path greedy decoder it would destroy the per-token logit. We therefore
compress phi through ``log(1+phi)`` so its scale matches the logit (a few
units), preserving the *qualitative* OPERA behaviour — penalise candidates
whose extension would form a stronger column attention pattern — without
the scale mismatch. Paper hyperparameters (sigma=50, alpha=1, k=8, Ncan=5)
remain unchanged.

This implementation costs ``ncan`` extra decode steps per token (≈ 4-5×
slower than vanilla greedy). The retrospection-rollback component is beam-
only and is intentionally omitted here so the comparison isolates OTP —
paper Ablation Table 5 shows OTP alone is the dominant contributor on
CHAIR.
"""
from __future__ import annotations

import copy
import math
from typing import List

import torch
from PIL import Image

from ..models.llava_wrapper import LlavaWrapper


def _kv_clone(pkv):
    try:
        return copy.deepcopy(pkv)
    except Exception:
        return None


def _phi_from_window(attn_window: torch.Tensor, sigma: float) -> float:
    """Eq. 5 — given a (W, W) lower-triangular attention window (no scale yet),
    return the maximum column product. Upper triangle is ignored."""
    W = attn_window.shape[0]
    if W < 2:
        return 0.0
    scaled = (attn_window * sigma).clamp(min=0.0)
    best = 0.0
    for c in range(W - 1):
        # product over rows i in (c+1, ..., W-1) along column c
        col = scaled[c + 1:, c]
        if col.numel() == 0:
            continue
        prod = float(torch.prod(col).item())
        if prod > best:
            best = prod
    return best


def _build_window(decoded_rows: List[torch.Tensor],
                  cand_row: torch.Tensor,
                  prefix_len: int,
                  t: int,
                  k_window: int) -> torch.Tensor:
    """Build a (W, W) attention window of the last W decoded text tokens
    INCLUDING the current candidate (position t, the W-th token in window).

    Each row in ``decoded_rows`` is a (1, 1, k_len_at_that_step) tensor where
    column ``prefix_len + j`` corresponds to decoded token j. The candidate's
    row is for position t — only its attention to decoded tokens
    [window_start .. t-1] is filled (the diagonal entry is zero because we
    don't apply self-loop)."""
    window_start = max(0, t - k_window + 1)
    W = t - window_start + 1                 # includes candidate row
    mat = torch.zeros(W, W, dtype=torch.float32, device=cand_row.device)
    for i in range(W):
        j_decoded = window_start + i
        if j_decoded < t:
            row = decoded_rows[j_decoded]
        else:
            row = cand_row                    # the candidate row
        # row shape: (1, 1, k_len)
        for j in range(i):                    # lower triangle only
            src_col = prefix_len + (window_start + j)
            if src_col < row.shape[-1]:
                mat[i, j] = row[0, 0, src_col].float()
    return mat


@torch.no_grad()
def opera_greedy_decode(wrapper: LlavaWrapper, image: Image.Image,
                        question: str, max_new_tokens: int = 64,
                        alpha: float = 1.0, sigma: float = 50.0,
                        k_window: int = 8, ncan: int = 5) -> str:
    # Model-agnostic input construction (LLaVA / Qwen2-VL handled by the
    # wrapper's own prepare_inputs + prefill; multimodal extras are stashed
    # by prepare_inputs and injected by prefill).
    enc = wrapper.prepare_inputs(image, question)
    attn = enc.get("attention_mask")

    # Prefill (no attention needed yet — we only need attention starting
    # from the first DECODED token's forward).
    out = wrapper.prefill(enc["input_ids"], enc["pixel_values"], attn)
    pkv = out.past_key_values
    logits = out.logits[:, -1, :]
    prefix_len = out.expanded_seq_len                  # incl. expanded <image>
    eos = int(wrapper.tokenizer.eos_token_id)

    decoded_tokens: List[int] = []
    decoded_attn_rows: List[torch.Tensor] = []         # last-layer max-head, (1,1,k_len)

    for step in range(max_new_tokens):
        # Top-Ncan candidates from base logit
        topk_vals, topk_idx = logits.topk(ncan, dim=-1)
        topk_vals = topk_vals[0]; topk_idx = topk_idx[0]
        t = step                                         # 0-indexed position of new token

        best_score = -float("inf")
        best_pick = None         # (token_id, attn_row, logits_after, pkv_after)

        for ci in range(ncan):
            cand_id = int(topk_idx[ci].item())
            cand = topk_idx[ci:ci + 1].view(1, 1)
            pkv_branch = _kv_clone(pkv)
            if pkv_branch is None:                       # fallback: skip candidate
                continue
            out_cand = wrapper.model(
                input_ids=cand,
                past_key_values=pkv_branch,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            # Last-layer attention, max over heads → (1, 1, k_len)
            last_attn = out_cand.attentions[-1]
            pooled = last_attn.max(dim=1).values         # (1, 1, k_len)

            window = _build_window(decoded_attn_rows, pooled,
                                   prefix_len, t, k_window)
            phi_raw = _phi_from_window(window, sigma)
            phi = math.log1p(phi_raw)         # greedy-scale adaptation
            score = float(topk_vals[ci].item()) - alpha * phi
            if score > best_score:
                best_score = score
                best_pick = (cand_id, pooled,
                             out_cand.logits[:, -1, :],
                             out_cand.past_key_values)

        if best_pick is None:                             # nothing usable
            break
        tok, attn_row, new_logits, new_pkv = best_pick
        if tok == eos:
            break
        decoded_tokens.append(tok)
        decoded_attn_rows.append(attn_row)
        logits = new_logits
        pkv = new_pkv

    return wrapper.tokenizer.decode(decoded_tokens, skip_special_tokens=True)
