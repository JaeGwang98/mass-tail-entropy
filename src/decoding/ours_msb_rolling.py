"""Rolling-attribution MSB — re-measure the LOO attribution at sentence boundaries.

Motivation
----------
MSB measures the LOO attribution once on the first L tokens, then locks the boost positions
for the entire generation.  This means later sentences (which often describe
different objects) keep getting boost on the wrong (early) segments.

Rolling MSB: at each sentence-ending token (".", "!", "?"), re-measure the LOO attribution
based on the next L tokens and update the boost positions.  Cap the number of
re-measurements to control cost.

Cost (1000 imgs CHAIR estimate):
  - Initial LOO attribution: K+1 = 7 forwards (~prefill cost)
  - Each re-measurement: 1 lookahead-prefill + K+1 occlusion forwards = ~8 prefills
  - With max_remeasures=2: ~16 extra prefill-equiv
  - Total: ~2x current MSB wall clock (~150 min instead of ~78 min)
"""

from __future__ import annotations

from typing import List

import torch
from PIL import Image

from ..models.llava_wrapper import LlavaWrapper
from ..utils.segmentation import PanopticSegmenter
from .ours_v3 import _greedy_lookahead
from .ours_v4 import (_segment_to_visual_token_indices,
                      _build_boost_mask, _build_step_boost_mask)
from .ours_msb import _topk_diverse, _shap_phis


def _is_sentence_end(tokenizer, token_id: int) -> bool:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    s = text.strip()
    if s in {".", "!", "?"}:
        return True
    # Some tokenizers merge punctuation with a space prefix (e.g. " .")
    return text.endswith((".", "!", "?"))


@torch.no_grad()
def _greedy_lookahead_until_period(wrapper, input_ids, pixel_values,
                                    attn_mask, max_steps: int = 32,
                                    min_steps: int = 4) -> List[int]:
    """Greedy generate up to the next sentence-ending token (capped at
    ``max_steps``).  Used for Rolling MSB re-measurement so the LOO attribution is computed
    over the FULL upcoming sentence rather than a fixed 8-token snippet.

    A min_steps floor avoids degenerate empty spans when the first generated
    token happens to be a period.
    """
    out = wrapper.prefill(input_ids, pixel_values, attn_mask)
    logits = out.logits[:, -1, :]
    pkv = out.past_key_values
    eos = int(wrapper.tokenizer.eos_token_id)
    span: List[int] = []
    for step in range(max_steps):
        nxt = logits.argmax(-1, keepdim=True)
        tok = int(nxt.item())
        if tok == eos:
            break
        span.append(tok)
        # Stop once we have enough tokens AND we just emitted a sentence end
        if step + 1 >= min_steps and _is_sentence_end(wrapper.tokenizer, tok):
            break
        logits, pkv = wrapper.decode_step(nxt, pkv)
        logits = logits[:, -1, :]
    return span


def _select_boost_positions(phis, segments, visual_pos, top_k, nms_iou):
    chosen = _topk_diverse(phis, segments, top_k, nms_iou)
    pos = []
    for ci in chosen:
        pos.extend(_segment_to_visual_token_indices(segments[ci].mask,
                                                    visual_pos))
    return sorted(set(pos))


@torch.no_grad()
def ours_msb_rolling_decode(wrapper: LlavaWrapper,
                            segmenter: PanopticSegmenter,
                            image: Image.Image, question: str,
                            max_new_tokens: int = 64,
                            boost_factor: float = 2.0,
                            top_k: int = 3,
                            lookahead: int = 8,
                            nms_iou: float = 0.5,
                            min_area_frac: float = 0.01,
                            max_segments: int = 6,
                            max_remeasures: int = 2,
                            re_lookahead_max: int = 32,
                            re_lookahead_min: int = 4) -> str:
    """Rolling-attribution MSB.

    Args:
        max_remeasures: max number of mid-generation LOO attribution re-measurements
            triggered by sentence-ending tokens.  0 reduces to vanilla MSB.
        re_lookahead_max: cap on tokens generated for each re-measurement
            lookahead (greedy until next period or this cap).
        re_lookahead_min: minimum tokens before a period stops the lookahead.
    """
    enc = wrapper.prepare_inputs(image, question)
    input_ids = enc["input_ids"]
    pixel_v = enc["pixel_values"]
    attn_mask = enc.get("attention_mask")
    seq_len = input_ids.shape[1]
    device, dtype = wrapper.device, wrapper.dtype
    eos = int(wrapper.tokenizer.eos_token_id)

    visual_pos = wrapper.visual_token_positions(input_ids)
    text_start = visual_pos[-1] + 1

    # Step 0: segmentation
    segments = segmenter.segment(image, min_area_frac=min_area_frac,
                                 max_segments=max_segments)
    if not segments:
        from .baseline import greedy_decode
        return greedy_decode(wrapper, image, question, max_new_tokens)

    # Step 1: initial lookahead
    span = _greedy_lookahead(wrapper, input_ids, pixel_v, attn_mask, lookahead)
    if not span:
        from .baseline import greedy_decode
        return greedy_decode(wrapper, image, question, max_new_tokens)

    # Step 2: initial LOO attribution
    phis = _shap_phis(wrapper, image, segments, input_ids, pixel_v, attn_mask,
                      span)
    boost_pos = _select_boost_positions(phis, segments, visual_pos, top_k,
                                        nms_iou)
    if not boost_pos:
        from .baseline import greedy_decode
        return greedy_decode(wrapper, image, question, max_new_tokens)

    # Step 3: boosted prefill
    boost_mask = _build_boost_mask(seq_len, boost_pos, text_start,
                                   boost_factor, device, dtype)
    out = wrapper.prefill(input_ids, pixel_v, boost_mask)
    logits = out.logits[:, -1, :]
    pkv = out.past_key_values

    generated: List[int] = []
    cur_len = seq_len
    n_remeasured = 0

    for _ in range(max_new_tokens):
        nxt = logits.argmax(-1, keepdim=True)
        token_id = int(nxt.item())
        if token_id == eos:
            break
        generated.append(token_id)

        # Sentence-boundary triggered re-measurement
        room_for_lookahead = max_new_tokens - len(generated) > re_lookahead_min + 4
        if (n_remeasured < max_remeasures and room_for_lookahead and
                _is_sentence_end(wrapper.tokenizer, token_id)):
            gen_t = torch.tensor([generated], device=device,
                                 dtype=input_ids.dtype)
            cur_input_ids = torch.cat([input_ids, gen_t], dim=1)
            if attn_mask is not None:
                cur_attn = torch.cat([attn_mask, torch.ones_like(gen_t)],
                                     dim=1)
            else:
                cur_attn = None
            new_span = _greedy_lookahead_until_period(
                wrapper, cur_input_ids, pixel_v, cur_attn,
                max_steps=re_lookahead_max, min_steps=re_lookahead_min)
            if new_span:
                new_phis = _shap_phis(wrapper, image, segments, cur_input_ids,
                                      pixel_v, cur_attn, new_span)
                new_pos = _select_boost_positions(new_phis, segments,
                                                  visual_pos, top_k, nms_iou)
                if new_pos:
                    boost_pos = new_pos
                    n_remeasured += 1

        step_mask = _build_step_boost_mask(cur_len, boost_pos, boost_factor,
                                           device, dtype)
        logits, pkv = wrapper.decode_step(nxt, pkv, attention_mask=step_mask)
        logits = logits[:, -1, :]
        cur_len += 1

    return wrapper.tokenizer.decode(generated, skip_special_tokens=True)
