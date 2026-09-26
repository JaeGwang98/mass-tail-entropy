"""SHAP-Attention Amplification (Ours, v4 — pilot).

Difference from v3:
  v3 builds a counterfactual image v' by mean-color masking the top-SHAP
  segment, then VCD-blends logits from v vs v'.  In greedy mode the small
  logit shift cannot move argmax through APC -> ours degenerates to baseline.

  v4 keeps v3's SHAP segment selection but instead of building v', it
  *amplifies the text->(top-SHAP visual tokens) attention* via a 4-D causal
  mask boost (+log gamma).  Single forward, no APC, no blend.  This makes
  the logit move at the score level (pre-softmax), which is robust to
  greedy decoding.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..models.llava_wrapper import LlavaWrapper
from ..utils.segmentation import PanopticSegmenter, Segment, mask_image_with_segment
from .ours_v3 import _greedy_lookahead, _shap_top_segment


# ---------------------------------------------------------------------------
# Map a (H, W) segment mask to visual-token indices in the 24x24 LLaVA grid.
# ---------------------------------------------------------------------------
def _segment_to_visual_token_indices(seg_mask: np.ndarray,
                                     visual_pos: Sequence[int],
                                     grid=24,
                                     occupancy_thresh: float = 0.3
                                     ) -> List[int]:
    """Return the absolute positions (in input_ids) of the visual tokens that
    overlap ``seg_mask`` by at least ``occupancy_thresh`` fraction.

    The H x W boolean mask is average-pooled down to the model's visual-token
    grid and thresholded. ``grid`` may be an int (square, e.g. 24 for
    LLaVA-1.5's 24x24=576 tokens) or a ``(grid_h, grid_w)`` tuple for models
    whose grid is non-square / image-dependent (e.g. Qwen2-VL). The flatten
    order is row-major, matching the patch-token order of both LLaVA and
    Qwen2-VL.
    """
    if isinstance(grid, (tuple, list)):
        gh, gw = int(grid[0]), int(grid[1])
    else:
        gh = gw = int(grid)
    if seg_mask.dtype != bool and seg_mask.dtype != np.bool_:
        seg_mask = seg_mask.astype(bool)
    t = torch.from_numpy(seg_mask.astype(np.float32))[None, None]    # (1,1,H,W)
    pooled = F.adaptive_avg_pool2d(t, output_size=(gh, gw))[0, 0]
    in_seg = (pooled >= occupancy_thresh).flatten()
    indices_local = torch.nonzero(in_seg, as_tuple=True)[0].tolist()
    return [visual_pos[i] for i in indices_local
            if i < len(visual_pos)]


# ---------------------------------------------------------------------------
# 4-D attention mask: standard causal + +log(gamma) boost at chosen positions
# ---------------------------------------------------------------------------
def _build_boost_mask(seq_len: int, boost_visual_pos: Sequence[int],
                      text_start: int, boost_factor: float,
                      device, dtype) -> torch.Tensor:
    mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)
    causal = torch.triu(torch.ones(seq_len, seq_len, device=device,
                                   dtype=torch.bool), diagonal=1)
    mask = mask.masked_fill(causal, float("-inf"))
    if boost_visual_pos and boost_factor > 1.0:
        idx = torch.tensor(boost_visual_pos, device=device, dtype=torch.long)
        boost = math.log(boost_factor)
        # Only text rows (positions >= text_start) get the boost.
        mask[..., text_start:, idx] = mask[..., text_start:, idx] + boost
    return mask


def _build_step_boost_mask(prefix_len: int,
                           boost_visual_pos: Sequence[int],
                           boost_factor: float,
                           device, dtype) -> torch.Tensor:
    """1-token-query mask used at each decode step."""
    new_len = prefix_len + 1
    mask = torch.zeros(1, 1, 1, new_len, device=device, dtype=dtype)
    if boost_visual_pos and boost_factor > 1.0:
        idx = torch.tensor(boost_visual_pos, device=device, dtype=torch.long)
        boost = math.log(boost_factor)
        mask[..., 0, idx] = mask[..., 0, idx] + boost
    return mask


# ---------------------------------------------------------------------------
# Top-level decoder
# ---------------------------------------------------------------------------
@torch.no_grad()
def ours_v4_decode(wrapper: LlavaWrapper, segmenter: PanopticSegmenter,
                   image: Image.Image, question: str,
                   max_new_tokens: int = 64,
                   boost_factor: float = 2.0,
                   lookahead: int = 1,
                   min_area_frac: float = 0.01,
                   max_segments: int = 6) -> str:
    enc = wrapper.prepare_inputs(image, question)
    input_ids = enc["input_ids"]
    pixel_v = enc["pixel_values"]
    attn_mask = enc.get("attention_mask")
    seq_len = input_ids.shape[1]
    device, dtype = wrapper.device, wrapper.dtype
    eos = int(wrapper.tokenizer.eos_token_id)

    visual_pos = wrapper.visual_token_positions(input_ids)
    text_start = visual_pos[-1] + 1

    # --- Step 0: segmentation ----------------------------------------------
    segments = segmenter.segment(image, min_area_frac=min_area_frac,
                                 max_segments=max_segments)

    if not segments:
        from .baseline import greedy_decode
        return greedy_decode(wrapper, image, question, max_new_tokens)

    # --- Step 1: lookahead -------------------------------------------------
    span = _greedy_lookahead(wrapper, input_ids, pixel_v, attn_mask, lookahead)
    if not span:
        from .baseline import greedy_decode
        return greedy_decode(wrapper, image, question, max_new_tokens)

    # --- Step 2: SHAP top segment ------------------------------------------
    i_star = _shap_top_segment(wrapper, image, segments, input_ids,
                               pixel_v, attn_mask, span)
    boost_pos = _segment_to_visual_token_indices(segments[i_star].mask,
                                                 visual_pos)
    if not boost_pos:
        from .baseline import greedy_decode
        return greedy_decode(wrapper, image, question, max_new_tokens)

    # --- Step 3: boosted prefill -------------------------------------------
    boost_mask = _build_boost_mask(seq_len, boost_pos, text_start,
                                   boost_factor, device, dtype)
    out = wrapper.prefill(input_ids, pixel_v, boost_mask)
    logits = out.logits[:, -1, :]
    pkv = out.past_key_values

    # --- Step 4: greedy decoding loop with rolling boost mask -------------
    generated: List[int] = []
    cur_len = seq_len
    for _ in range(max_new_tokens):
        nxt = logits.argmax(-1, keepdim=True)
        token_id = int(nxt.item())
        if token_id == eos:
            break
        generated.append(token_id)
        step_mask = _build_step_boost_mask(cur_len, boost_pos,
                                           boost_factor, device, dtype)
        logits, pkv = wrapper.decode_step(nxt, pkv, attention_mask=step_mask)
        logits = logits[:, -1, :]
        cur_len += 1
    return wrapper.tokenizer.decode(generated, skip_special_tokens=True)
