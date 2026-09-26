"""Prior-calibrated decoding (SBC "flat" arm).

Rationale
---------
On closed-form QA (POPE) the dominant hallucination is the *language prior*
(yes-bias), not SHAP over-concentration — see ``HALLU_DISTRIBUTION.md``.  The
SHAP attribution for the answer token is essentially flat in that regime: no
segment moves ``log p(yes)``.  When there is no decisive visual evidence the
model falls back on the prior.

This decoder cancels that prior term by contrasting against a *blank* image —
``torch.zeros_like(pixel_values)``, which (after CLIP normalisation) is a
neutral mid-gray rectangle carrying no scene information.  It is the
zero-information limit of VCD's diffusion-noise corruption (cf. M3ID / VDD):

    logit_final = (1 + alpha) * logit(y | x, v) - alpha * logit(y | x, blank)

with VCD's adaptive plausibility constraint

    V_head = { y : p(y | x, v) >= beta * max_w p(w | x, v) } .

Cost: 2 prefills + 1 forward / token (same as VCD), no segmenter.

In the full SBC pipeline this is applied *only* when the per-token SHAP
attribution is flat (over-dispersed); used standalone here for the POPE pilot
(POPE answers are 1-token, so the SHAP gate would always fire anyway).
"""

from __future__ import annotations

from typing import List

import torch
from PIL import Image

from ..models.llava_wrapper import LlavaWrapper
from .vcd import _apc_filter


@torch.no_grad()
def ours_pmi_decode(wrapper: LlavaWrapper, image: Image.Image, question: str,
                    max_new_tokens: int = 64,
                    alpha: float = 1.0, beta: float = 0.1,
                    sampling: str = "greedy") -> str:
    """Blank-image prior-calibrated contrastive decoding.

    Args:
        alpha: contrast strength.  alpha=1.0 mirrors VCD's default; the blank
            image is a stronger perturbation than diffusion noise, so 0.5 is a
            sensible conservative alternative.
        beta: adaptive plausibility cutoff (same role as in VCD).
        sampling: "greedy" (default) reproduces the deterministic protocol used
            by our other ``ours_*`` methods; "direct" matches VCD Tab. 1.
    """
    enc = wrapper.prepare_inputs(image, question)
    input_ids = enc["input_ids"]
    pixel_v = enc["pixel_values"]
    attn_mask = enc.get("attention_mask")
    # Blank = all-zeros in the *normalised* pixel space == per-channel CLIP
    # mean == featureless mid-gray.  Carries no scene content -> isolates the
    # text-only language prior.
    pixel_blank = torch.zeros_like(pixel_v)

    out_v = wrapper.prefill(input_ids, pixel_v, attn_mask)
    out_b = wrapper.prefill(input_ids, pixel_blank, attn_mask)

    logits_v = out_v.logits[:, -1, :]
    logits_b = out_b.logits[:, -1, :]
    pkv_v, pkv_b = out_v.past_key_values, out_b.past_key_values
    eos = int(wrapper.tokenizer.eos_token_id)

    generated: List[int] = []
    for _ in range(max_new_tokens):
        logit_blend = (1.0 + alpha) * logits_v - alpha * logits_b
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
        logits_b, pkv_b = wrapper.decode_step(nxt, pkv_b)
        logits_v = logits_v[:, -1, :]
        logits_b = logits_b[:, -1, :]
    return wrapper.tokenizer.decode(generated, skip_special_tokens=True)
