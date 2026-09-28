"""Attribution-Targeted Visual Contrastive Decoding (Ours, v3).

Pipeline (mirrors 방법론.md):
  Step 0:  Mask2Former-Swin-L panoptic segmentation -> up to K segments.
  Step 1:  Lookahead - greedy first L tokens given (x, v).
  Step 2:  Per-segment LOO attribution via leave-one-out occlusion (mean-color fill):
              phi_i = log p(span | x, v) - log p(span | x, v \\ s_i)
           batched as K parallel forwards.
  Step 3:  v' = v with segment s_{i*} masked (i* = argmax phi).
  Step 4:  Per-token VCD blend with two parallel KV-caches and APC.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from PIL import Image

from ..models.llava_wrapper import LlavaWrapper
from ..utils.segmentation import (PanopticSegmenter, mask_image_with_segment,
                                  Segment)
from .vcd import _apc_filter


# ---------------------------------------------------------------------------
# Step 1+2: greedy lookahead + per-segment LOO attribution
# ---------------------------------------------------------------------------
@torch.no_grad()
def _greedy_lookahead(wrapper: LlavaWrapper, input_ids, pixel_values,
                      attn_mask, n_steps: int, **extra) -> List[int]:
    out = wrapper.prefill(input_ids, pixel_values, attn_mask, **extra)
    logits = out.logits[:, -1, :]
    pkv = out.past_key_values
    eos = int(wrapper.tokenizer.eos_token_id)
    span: List[int] = []
    for _ in range(n_steps):
        nxt = logits.argmax(-1, keepdim=True)
        tok = int(nxt.item())
        if tok == eos:
            break
        span.append(tok)
        logits, pkv = wrapper.decode_step(nxt, pkv)
        logits = logits[:, -1, :]
    return span


@torch.no_grad()
def _logp_span_under_image(wrapper: LlavaWrapper,
                           prompt_ids: torch.Tensor,
                           pixel_values: torch.Tensor,
                           attn_mask: torch.Tensor,
                           span: Sequence[int],
                           **extra) -> float:
    """Teacher-forced log p(span | prompt, pixel_values).
    ``extra`` is forwarded to ``wrapper.model`` (e.g. ``image_grid_thw`` for
    Qwen2-VL)."""
    if not span:
        return 0.0
    span_tensor = torch.tensor([list(span)], device=prompt_ids.device,
                               dtype=prompt_ids.dtype)
    full_ids = torch.cat([prompt_ids, span_tensor], dim=1)
    if attn_mask is not None:
        full_attn = torch.cat([attn_mask,
                               torch.ones_like(span_tensor)], dim=1)
    else:
        full_attn = None
    # Qwen2-VL: extend mm_token_type_ids by len(span) zeros (text type)
    extra = dict(extra)
    if "mm_token_type_ids" in extra:
        mtt = extra["mm_token_type_ids"]
        extra["mm_token_type_ids"] = torch.cat(
            [mtt, torch.zeros_like(span_tensor)], dim=1)
    out = wrapper.model(input_ids=full_ids, pixel_values=pixel_values,
                        attention_mask=full_attn, use_cache=False,
                        return_dict=True, **extra)
    logits = out.logits.float()                            # (1, S, V)
    # The token at expanded position p is predicted from logits[p-1].
    # Locate where the span begins in the *expanded* sequence: it always
    # follows the prompt (which expands to 575 extra image tokens).  The
    # span lives at the very end, so we can index from the right.
    L = len(span)
    span_logits = logits[0, -L - 1:-1, :]                  # (L, V)
    log_probs = torch.log_softmax(span_logits, dim=-1)
    span_t = torch.tensor(span, device=log_probs.device, dtype=torch.long)
    return float(log_probs[range(L), span_t].sum().item())


def _shap_top_segment(wrapper: LlavaWrapper, image: Image.Image,
                      segments: Sequence[Segment], prompt_ids: torch.Tensor,
                      pixel_values_v: torch.Tensor,
                      attn_mask: torch.Tensor, span: Sequence[int]) -> int:
    """Return the index of the segment with the largest LOO drop."""
    base_lp = _logp_span_under_image(wrapper, prompt_ids, pixel_values_v,
                                     attn_mask, span)
    best_phi, best_i = -float("inf"), 0
    for i, seg in enumerate(segments):
        masked_img = mask_image_with_segment(image, seg.mask)
        # Re-process to obtain pixel_values_i (same shape as pixel_values_v).
        # Use image_processor directly so we don't need to repass text.
        enc_i = wrapper.processor.image_processor(images=masked_img,
                                                  return_tensors="pt")
        pix_i = enc_i["pixel_values"].to(pixel_values_v.device,
                                         dtype=pixel_values_v.dtype)
        lp_i = _logp_span_under_image(wrapper, prompt_ids, pix_i,
                                      attn_mask, span)
        phi = base_lp - lp_i
        if phi > best_phi:
            best_phi, best_i = phi, i
    return best_i


# ---------------------------------------------------------------------------
# Top-level decoding entry point
# ---------------------------------------------------------------------------
@torch.no_grad()
def ours_v3_decode(wrapper: LlavaWrapper, segmenter: PanopticSegmenter,
                   image: Image.Image, question: str,
                   max_new_tokens: int = 64,
                   alpha: float = 1.0, beta: float = 0.1,
                   lookahead: int = 1,
                   sampling: str = "greedy",
                   min_area_frac: float = 0.01,
                   max_segments: int = 6) -> str:
    enc = wrapper.prepare_inputs(image, question)
    input_ids = enc["input_ids"]
    pixel_v = enc["pixel_values"]
    attn_mask = enc.get("attention_mask")

    # ---- Step 0: segmentation ---------------------------------------------
    segments = segmenter.segment(image, min_area_frac=min_area_frac,
                                 max_segments=max_segments)

    # If no usable segment exists, fall back to greedy on (x, v).
    if not segments:
        from .baseline import greedy_decode
        return greedy_decode(wrapper, image, question, max_new_tokens)

    # ---- Step 1: lookahead -------------------------------------------------
    span = _greedy_lookahead(wrapper, input_ids, pixel_v, attn_mask, lookahead)
    if not span:
        from .baseline import greedy_decode
        return greedy_decode(wrapper, image, question, max_new_tokens)

    # ---- Step 2: per-segment LOO attribution -----------------------------------------
    i_star = _shap_top_segment(wrapper, image, segments, input_ids,
                               pixel_v, attn_mask, span)

    # ---- Step 3: build counterfactual v' ----------------------------------
    masked_img = mask_image_with_segment(image, segments[i_star].mask)
    enc_p = wrapper.processor.image_processor(images=masked_img,
                                              return_tensors="pt")
    pixel_vp = enc_p["pixel_values"].to(pixel_v.device, dtype=pixel_v.dtype)

    # ---- Step 4: VCD blend with two parallel KV-caches --------------------
    out_v = wrapper.prefill(input_ids, pixel_v, attn_mask)
    out_vp = wrapper.prefill(input_ids, pixel_vp, attn_mask)
    logits_v = out_v.logits[:, -1, :]
    logits_vp = out_vp.logits[:, -1, :]
    pkv_v, pkv_vp = out_v.past_key_values, out_vp.past_key_values
    eos = int(wrapper.tokenizer.eos_token_id)

    generated: List[int] = []
    for _ in range(max_new_tokens):
        logit_blend = (1.0 + alpha) * logits_v - alpha * logits_vp
        keep = _apc_filter(logits_v, beta)
        logit_blend = logit_blend.masked_fill(~keep, float("-inf"))

        if sampling == "greedy":
            nxt = logit_blend.argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(logit_blend.float(), dim=-1)
            if torch.isnan(probs).any() or probs.sum() == 0:
                nxt = logits_v.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(probs, num_samples=1)
        token_id = int(nxt.item())
        if token_id == eos:
            break
        generated.append(token_id)
        logits_v, pkv_v = wrapper.decode_step(nxt, pkv_v)
        logits_vp, pkv_vp = wrapper.decode_step(nxt, pkv_vp)
        logits_v = logits_v[:, -1, :]
        logits_vp = logits_vp[:, -1, :]
    return wrapper.tokenizer.decode(generated, skip_special_tokens=True)
