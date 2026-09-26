"""Baseline decoding strategies.

* ``greedy_decode`` – used by the SAE CHAIR protocol (Tab. 1, max_len=512,
  greedy).
* ``sample_decode`` – direct post-softmax sampling, used by VCD POPE
  (called "Regular" in Tab. 1 of Leng et al., CVPR 2024).
"""

from __future__ import annotations

from typing import List

import torch
from PIL import Image

from ..models.llava_wrapper import LlavaWrapper


def _stop_id(wrapper: LlavaWrapper) -> int:
    return int(wrapper.tokenizer.eos_token_id)


def _extra_from_enc(enc):
    return {k: enc[k] for k in ("image_grid_thw", "mm_token_type_ids")
            if k in enc}


@torch.no_grad()
def greedy_decode(wrapper: LlavaWrapper, image: Image.Image, question: str,
                  max_new_tokens: int = 64) -> str:
    enc = wrapper.prepare_inputs(image, question)
    out = wrapper.prefill(enc["input_ids"], enc["pixel_values"],
                          enc.get("attention_mask"), **_extra_from_enc(enc))
    logits = out.logits[:, -1, :]
    pkv = out.past_key_values
    eos = _stop_id(wrapper)

    generated: List[int] = []
    for _ in range(max_new_tokens):
        nxt = logits.argmax(-1, keepdim=True)
        token_id = int(nxt.item())
        if token_id == eos:
            break
        generated.append(token_id)
        logits, pkv = wrapper.decode_step(nxt, pkv)
        logits = logits[:, -1, :]
    return wrapper.tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def sample_decode(wrapper: LlavaWrapper, image: Image.Image, question: str,
                  max_new_tokens: int = 64) -> str:
    """Post-softmax direct sampling, the 'Regular' decoding column in VCD."""
    enc = wrapper.prepare_inputs(image, question)
    out = wrapper.prefill(enc["input_ids"], enc["pixel_values"],
                          enc.get("attention_mask"), **_extra_from_enc(enc))
    logits = out.logits[:, -1, :]
    pkv = out.past_key_values
    eos = _stop_id(wrapper)

    generated: List[int] = []
    for _ in range(max_new_tokens):
        probs = torch.softmax(logits.float(), dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        token_id = int(nxt.item())
        if token_id == eos:
            break
        generated.append(token_id)
        logits, pkv = wrapper.decode_step(nxt, pkv)
        logits = logits[:, -1, :]
    return wrapper.tokenizer.decode(generated, skip_special_tokens=True)
