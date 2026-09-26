"""Visual Contrastive Decoding (VCD).

Implements Eqs. 3–5 of Leng et al., CVPR 2024:

    logits_blend = (1 + alpha) * logit(y | x, v) - alpha * logit(y | x, v')

with the adaptive plausibility constraint

    V_head = { y : p(y | x, v) >= beta * max_w p(w | x, v) }.

The distorted image v' is produced by adding ``noise_step`` of forward
diffusion noise as in the official repo.
"""

from __future__ import annotations

from typing import List

import torch
from PIL import Image

from ..models.llava_wrapper import LlavaWrapper
from ..utils.distortion import add_diffusion_noise


def _apc_filter(logits_v: torch.Tensor, beta: float) -> torch.Tensor:
    """Adaptive plausibility constraint (Eq. 4).  Returns a boolean keep-mask."""
    probs = torch.softmax(logits_v.float(), dim=-1)
    threshold = beta * probs.max(dim=-1, keepdim=True).values
    return probs >= threshold


@torch.no_grad()
def vcd_decode(wrapper: LlavaWrapper, image: Image.Image, question: str,
               max_new_tokens: int = 64,
               alpha: float = 1.0, beta: float = 0.1,
               noise_step: int = 999,
               sampling: str = "direct") -> str:
    """``sampling='direct'`` reproduces VCD Tab. 1 (post-softmax sampling)."""
    enc = wrapper.prepare_inputs(image, question)
    pixel_v = enc["pixel_values"]
    pixel_vp = add_diffusion_noise(pixel_v.clone(), noise_step=noise_step)

    # Two parallel prefills
    out_v = wrapper.prefill(enc["input_ids"], pixel_v, enc.get("attention_mask"))
    out_vp = wrapper.prefill(enc["input_ids"], pixel_vp, enc.get("attention_mask"))

    logits_v = out_v.logits[:, -1, :]
    logits_vp = out_vp.logits[:, -1, :]
    pkv_v, pkv_vp = out_v.past_key_values, out_vp.past_key_values
    eos = int(wrapper.tokenizer.eos_token_id)

    generated: List[int] = []
    for _ in range(max_new_tokens):
        # contrastive blend (Eq. 3)
        logit_blend = (1.0 + alpha) * logits_v - alpha * logits_vp
        # APC filter (Eq. 4)
        keep = _apc_filter(logits_v, beta)
        logit_blend = logit_blend.masked_fill(~keep, float("-inf"))

        if sampling == "greedy":
            nxt = logit_blend.argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(logit_blend.float(), dim=-1)
            # numerical safety: APC may have killed everything in float16 land
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
