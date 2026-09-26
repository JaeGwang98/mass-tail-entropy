"""Adaptive Information Flow (AIF, Liu et al., 2026).

The implementation follows Sec. 4 of the paper:

1. Run one forward pass with ``output_attentions=True`` to obtain layer-wise
   attentions.  For each visual token v_i compute

       d^l_{v_i} = max_j  mean_h  a^{l,h}_{j, i}

   where j ranges over the text tokens that come after the visual block.
2. Token entropy

       Ent_{v_i} = sum_l - p^l log p^l ,  p^l = d^l_{v_i} / (L * mu_{v_i}) .

   High Ent => irregular activation => unimportant token (masked first).
3. Adaptive mask-ratio search.  S_0 = entropy of the mu distribution; for
   each candidate ratio mask the high-Ent tokens and recompute the entropy
   on the *retained* set; pick the ratio whose entropy is most distant from
   S_0.
4. Re-run decoding with a 4-D causal mask in which text positions cannot
   attend to the masked visual positions (visual<->visual stays open).
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from PIL import Image

from ..models.llava_wrapper import LlavaWrapper


# ---------------------------------------------------------------------------
# Step 1+2: token-dynamics entropy
# ---------------------------------------------------------------------------
def _token_entropy(attentions: Sequence[torch.Tensor],
                   visual_pos: Sequence[int],
                   text_start: int) -> torch.Tensor:
    """Returns ``Ent_{v_i}`` of shape (n_visual,)."""
    L = len(attentions)
    # d_per_layer[l, i] = max_j  mean_h  attn[l, h, j, visual_i]
    d_layers = []
    for layer_attn in attentions:                    # (1, H, S, S)
        attn = layer_attn[0].mean(dim=0).float()      # (S, S), avg over heads
        text_to_vis = attn[text_start:, visual_pos]   # (T, n_vis)
        d_layers.append(text_to_vis.max(dim=0).values)  # (n_vis,)
    d = torch.stack(d_layers, dim=0)                  # (L, n_vis)
    mu = d.mean(dim=0).clamp(min=1e-12)               # (n_vis,)
    p = d / (L * mu)                                  # (L, n_vis), sums to 1 over L
    p = p.clamp(min=1e-12)
    ent = -(p * p.log()).sum(dim=0)                   # (n_vis,)
    return ent, mu


# ---------------------------------------------------------------------------
# Step 3: adaptive mask-ratio search (Eq. 5)
# ---------------------------------------------------------------------------
def _shannon(p: torch.Tensor) -> float:
    p = p[p > 0]
    return float((-(p * p.log())).sum().item())


def _select_mask_ratio(ent: torch.Tensor, mu: torch.Tensor,
                       ratios: Sequence[float],
                       max_ratio: float = 0.5) -> float:
    """Adaptive search per AIF Sec. 4.3.

    Note: with LLaVA-1.5-7B the raw "most distant from S0" criterion
    saturates at r=0.9 on every image, which masks 90% of visual tokens
    and destroys answer quality.  We restrict the search to ratios <=
    ``max_ratio`` (the AIF ablation Tab. 7 also compares "under the same
    masking ratio", implying their final pipeline does not use the very
    aggressive end of the grid).
    """
    p_full = mu / mu.sum()
    s0 = _shannon(p_full)
    order = torch.argsort(ent, descending=True)         # high-Ent first
    best_ratio, best_dist = ratios[0], -1.0
    n = ent.numel()
    for r in ratios:
        if r > max_ratio:
            continue
        n_mask = max(1, int(round(r * n)))
        keep = order[n_mask:]                            # complement
        if keep.numel() == 0:
            continue
        mu_keep = mu[keep]
        if mu_keep.sum() <= 0:
            continue
        s_r = _shannon(mu_keep / mu_keep.sum())
        d = abs(s_r - s0)
        if d > best_dist:
            best_dist, best_ratio = d, r
    return best_ratio


# ---------------------------------------------------------------------------
# 4-D causal-mask construction
# ---------------------------------------------------------------------------
def _build_blocked_mask(seq_len: int, blocked_visual_pos: Sequence[int],
                        text_start: int, device, dtype) -> torch.Tensor:
    mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)
    causal = torch.triu(torch.ones(seq_len, seq_len, device=device,
                                   dtype=torch.bool), diagonal=1)
    mask = mask.masked_fill(causal, float("-inf"))
    if blocked_visual_pos:
        idx = torch.tensor(blocked_visual_pos, device=device, dtype=torch.long)
        # text rows (text_start..seq_len-1) cannot attend to blocked visual cols
        mask[..., text_start:, idx] = float("-inf")
    return mask


def _build_step_mask(prefix_len: int, blocked_visual_pos: Sequence[int],
                     device, dtype) -> torch.Tensor:
    """Mask for the *new* (1-token) query: shape (1, 1, 1, prefix_len + 1)."""
    new_len = prefix_len + 1
    mask = torch.zeros(1, 1, 1, new_len, device=device, dtype=dtype)
    if blocked_visual_pos:
        idx = torch.tensor(blocked_visual_pos, device=device, dtype=torch.long)
        mask[..., 0, idx] = float("-inf")
    return mask


# ---------------------------------------------------------------------------
# Top-level decoding entry point
# ---------------------------------------------------------------------------
@torch.no_grad()
def aif_decode(wrapper: LlavaWrapper, image: Image.Image, question: str,
               max_new_tokens: int = 64,
               sampling: str = "greedy",
               mask_ratios: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5,
                                               0.6, 0.7, 0.8, 0.9),
               max_ratio: float = 0.5) -> str:
    enc = wrapper.prepare_inputs(image, question)
    input_ids = enc["input_ids"]
    pixel_values = enc["pixel_values"]
    seq_len = input_ids.shape[1]
    device, dtype = wrapper.device, wrapper.dtype

    # --- 1. Probe forward with attentions ---------------------------------
    out = wrapper.prefill(input_ids, pixel_values, enc.get("attention_mask"),
                          output_attentions=True)
    visual_pos = wrapper.visual_token_positions(input_ids)
    text_start = visual_pos[-1] + 1
    ent, mu = _token_entropy(out.attentions, visual_pos, text_start)
    del out

    # --- 2. Mask selection -------------------------------------------------
    r_star = _select_mask_ratio(ent, mu, mask_ratios, max_ratio=max_ratio)
    n_mask = max(1, int(round(r_star * ent.numel())))
    order = torch.argsort(ent, descending=True)
    blocked_local = order[:n_mask].tolist()
    blocked_pos = [visual_pos[i] for i in blocked_local]

    # --- 3. Prefill with modulated mask ------------------------------------
    mask4d = _build_blocked_mask(seq_len, blocked_pos, text_start, device, dtype)
    out2 = wrapper.prefill(input_ids, pixel_values, mask4d)
    logits = out2.logits[:, -1, :]
    pkv = out2.past_key_values

    # --- 4. Decoding loop with rolling 4-D mask ----------------------------
    eos = int(wrapper.tokenizer.eos_token_id)
    generated: List[int] = []
    cur_len = seq_len
    for _ in range(max_new_tokens):
        if sampling == "greedy":
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(logits.float(), dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        token_id = int(nxt.item())
        if token_id == eos:
            break
        generated.append(token_id)
        step_mask = _build_step_mask(cur_len, blocked_pos, device, dtype)
        logits, pkv = wrapper.decode_step(nxt, pkv, attention_mask=step_mask)
        logits = logits[:, -1, :]
        cur_len += 1
    return wrapper.tokenizer.decode(generated, skip_special_tokens=True)
