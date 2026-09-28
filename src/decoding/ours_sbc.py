r"""SBC — Segment-Based Conditioning  (v3 gate).

One LOO attribution measurement, one principled rule (no baseline / "do-nothing" route,
no benchmark detection):

  PMI  if  H >= tau_mid  AND  span(v) == span(blank)
           (LOO attribution is flat  AND  the image-free answer is identical
            -> the prediction is prior-driven)
  MSB  otherwise
           (redistribute attention over the top-k attribution segments — behaves like
            gate-free MSB-sent, which is what works on free-form captioning)

  greedy fallback if < 2 segments or no usable visual tokens.

Why this works on both: a caption's first sentence never matches the
blank-image caption -> always MSB -> equals gate-free MSB-sent (CHAIR).  A
closed-form yes/no answer that the model gives even with a blank image -> PMI
-> equals ours_pmi (POPE).  The blank-agreement test, not span length, is the
free-form/closed-form discriminator, so this is one rule rather than per-task
logic.

Cost: 1 (lookahead) + K (occlusion) prefills, then either
  - MSB: 1 boosted prefill + greedy gen, or
  - PMI: 1 blank prefill (the blank-agreement check) + a 2-cache contrastive
    gen that *reuses* the lookahead prefill (real arm) and the blank prefill
    (blank arm) — so PMI adds no extra prefills beyond the blank-agreement one.
"""
from __future__ import annotations

import copy
import math
from typing import List

import numpy as np
import torch
from PIL import Image

from ..models.llava_wrapper import LlavaWrapper
from ..utils.segmentation import PanopticSegmenter
from .ours_msb_rolling import _is_sentence_end
from .ours_msb import _shap_phis_batched as _shap_phis, _topk_diverse
from .ours_v4 import (_segment_to_visual_token_indices,
                      _build_boost_mask, _build_step_boost_mask)
from .vcd import _apc_filter

_MIN_STEPS = 4   # mirrors ours_msb_rolling._greedy_lookahead_until_period


def _norm_entropy(phi: np.ndarray) -> float:
    """Normalized entropy of softmax(phi) in [0, 1].  K=1 -> 1.0 (degenerate)."""
    k = len(phi)
    if k <= 1:
        return 1.0
    z = phi - phi.max()
    p = np.exp(z)
    p = p / p.sum()
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / math.log(k))


def _kv_clone(pkv):
    """Deep-copy a transformers Cache / legacy tuple so that later decode steps
    on the *original* don't mutate this snapshot.  Returns ``None`` if the
    object can't be copied (the caller then just re-runs the prefill)."""
    try:
        return copy.deepcopy(pkv)
    except Exception:
        return None


@torch.no_grad()
def _lookahead_with_logp(wrapper, input_ids, pixel_values, attn_mask,
                         use_sentence: bool, lookahead: int, max_steps: int = 32):
    """Greedy lookahead that also returns the teacher-forced log p(span | x, v)
    accumulated step-by-step (so the attribution need not re-forward the original image)
    *and* a pristine ``(prompt_logits, prompt_pkv)`` snapshot of the prefill at
    the prompt end — PMI generation uses the same unboosted prefill for its
    real-image arm, so it can reuse this instead of recomputing it."""
    out = wrapper.prefill(input_ids, pixel_values, attn_mask)
    prompt_logits = out.logits[:, -1, :].clone()
    prompt_pkv = _kv_clone(out.past_key_values)
    logits = out.logits[:, -1, :]
    pkv = out.past_key_values
    eos = int(wrapper.tokenizer.eos_token_id)
    span: List[int] = []
    base_lp = 0.0
    n = max_steps if use_sentence else lookahead
    for step in range(n):
        lp_row = torch.log_softmax(logits.float(), dim=-1)
        nxt = logits.argmax(-1, keepdim=True)
        tok = int(nxt.item())
        if tok == eos:
            break
        base_lp += float(lp_row[0, tok].item())
        span.append(tok)
        if (use_sentence and step + 1 >= _MIN_STEPS
                and _is_sentence_end(wrapper.tokenizer, tok)):
            break
        logits, pkv = wrapper.decode_step(nxt, pkv)
        logits = logits[:, -1, :]
    return span, base_lp, prompt_logits, prompt_pkv


def _norm_span_text(tokenizer, span):
    """Normalize a decoded span for the blank-agreement comparison: the test
    asks 'is the answer the same with vs. without the image?' — a *semantic*
    question, not a token-identity one.  Exact token-span equality is brittle
    across tokenizers/chat-templates (e.g. Qwen2-VL emits 'No' for a blank
    image but 'no'/'yes' when grounded — same answer, different tokens, so the
    old exact match never fired and PMI never routed).  Lower-casing + stripping
    non-alphanumerics makes prior-driven binary answers compare equal while
    long free-form captions (CHAIR) still differ → still route MSB."""
    txt = tokenizer.decode(list(span), skip_special_tokens=True)
    return "".join(ch for ch in txt.lower() if ch.isalnum())


@torch.no_grad()
def _blank_span_matches(wrapper, input_ids, pixel_blank, attn_mask, target_span,
                        use_sentence: bool, lookahead: int, max_steps: int = 32):
    """Returns ``(matched, prompt_logits, prompt_pkv)``.

    ``matched`` is True iff greedily decoding under the blank image yields the
    *same normalized answer* as ``target_span``.  Comparison is on
    case/punctuation-normalized text rather than exact token ids so the gate
    is tokenizer-agnostic (see ``_norm_span_text``).  ``(prompt_logits,
    prompt_pkv)`` is a pristine snapshot of the blank prefill at the prompt
    end, reused by PMI generation's blank arm."""
    out = wrapper.prefill(input_ids, pixel_blank, attn_mask)
    prompt_logits = out.logits[:, -1, :].clone()
    prompt_pkv = _kv_clone(out.past_key_values)
    if not target_span:
        return False, prompt_logits, prompt_pkv
    logits = out.logits[:, -1, :]
    pkv = out.past_key_values
    eos = int(wrapper.tokenizer.eos_token_id)
    n = max_steps if use_sentence else lookahead
    blank_span = []
    for step in range(n):
        tok = int(logits.argmax(-1).item())
        if tok == eos:
            break
        blank_span.append(tok)
        if (use_sentence and step + 1 >= _MIN_STEPS
                and _is_sentence_end(wrapper.tokenizer, tok)):
            break
        nxt = torch.tensor([[tok]], device=input_ids.device,
                           dtype=input_ids.dtype)
        logits, pkv = wrapper.decode_step(nxt, pkv)
        logits = logits[:, -1, :]
    matched = (_norm_span_text(wrapper.tokenizer, blank_span) ==
               _norm_span_text(wrapper.tokenizer, target_span))
    return matched, prompt_logits, prompt_pkv


@torch.no_grad()
@torch.no_grad()
def _attn_segment_mass(wrapper, input_ids, pixel_v, attn, segments) -> np.ndarray:
    """E5 (attention-ranked MSB): score each panoptic segment by the
    last-layer, head-averaged attention mass that the final prompt token
    places on the segment's visual tokens. One extra prefill, no occlusion
    passes. Used only as a *ranking* signal for MSB's top-K selection."""
    out_a = wrapper.prefill(input_ids, pixel_v, attn, output_attentions=True)
    last = out_a.attentions[-1][0].mean(dim=0)[-1].float()
    vis_pos = wrapper.visual_token_positions(input_ids)
    try:
        grid = wrapper.visual_grid(input_ids)
    except TypeError:
        grid = wrapper.visual_grid()
    scores = []
    for seg in segments:
        idx = _segment_to_visual_token_indices(seg.mask, vis_pos, grid=grid)
        scores.append(float(last[idx].sum().item()) if idx else 0.0)
    del out_a
    return np.asarray(scores, dtype=np.float64)


def _generate_msb(wrapper, input_ids, pixel_v, attn, segments, phis,
                  top_k, nms_iou, boost_factor, max_new_tokens) -> str:
    visual_pos = wrapper.visual_token_positions(input_ids)
    text_start = visual_pos[-1] + 1
    seq_len = input_ids.shape[1]
    device, dtype = wrapper.device, wrapper.dtype
    eos = int(wrapper.tokenizer.eos_token_id)

    grid = wrapper.visual_grid(input_ids)
    chosen = _topk_diverse(phis, segments, top_k, nms_iou)
    boost_pos: List[int] = []
    for ci in chosen:
        boost_pos.extend(_segment_to_visual_token_indices(segments[ci].mask,
                                                          visual_pos,
                                                          grid=grid))
    boost_pos = sorted(set(boost_pos))
    if not boost_pos:
        return None  # no usable visual tokens — caller falls back to greedy

    bm = _build_boost_mask(seq_len, boost_pos, text_start, boost_factor,
                           device, dtype)
    out = wrapper.prefill(input_ids, pixel_v, bm)
    logits = out.logits[:, -1, :]
    pkv = out.past_key_values
    gen: List[int] = []
    cur = seq_len
    for _ in range(max_new_tokens):
        nxt = logits.argmax(-1, keepdim=True)
        tid = int(nxt.item())
        if tid == eos:
            break
        gen.append(tid)
        sm = _build_step_boost_mask(cur, boost_pos, boost_factor, device, dtype)
        logits, pkv = wrapper.decode_step(nxt, pkv, attention_mask=sm)
        logits = logits[:, -1, :]
        cur += 1
    return wrapper.tokenizer.decode(gen, skip_special_tokens=True)


@torch.no_grad()
def _generate_pmi(wrapper, input_ids, pixel_v, attn, alpha, beta,
                  max_new_tokens, v_prefill=None, b_prefill=None) -> str:
    """Two-cache contrastive generation: logit = (1+alpha)*logit(y|v) -
    alpha*logit(y|blank), with VCD's APC.  ``v_prefill`` / ``b_prefill``, if
    supplied as ``(prompt_logits, prompt_pkv)`` with a non-None pkv, are reused
    in place of re-running the (identical) real-image / blank-image prefill."""
    eos = int(wrapper.tokenizer.eos_token_id)
    if v_prefill is not None and v_prefill[1] is not None:
        logits_v, pkv_v = v_prefill
    else:
        out_v = wrapper.prefill(input_ids, pixel_v, attn)
        logits_v, pkv_v = out_v.logits[:, -1, :], out_v.past_key_values
    if b_prefill is not None and b_prefill[1] is not None:
        logits_b, pkv_b = b_prefill
    else:
        out_b = wrapper.prefill(input_ids, torch.zeros_like(pixel_v), attn)
        logits_b, pkv_b = out_b.logits[:, -1, :], out_b.past_key_values
    gen: List[int] = []
    for _ in range(max_new_tokens):
        blend = (1.0 + alpha) * logits_v - alpha * logits_b
        keep = _apc_filter(logits_v, beta)
        blend = blend.masked_fill(~keep, float("-inf"))
        nxt = blend.argmax(-1, keepdim=True)
        tid = int(nxt.item())
        if tid == eos:
            break
        gen.append(tid)
        logits_v, pkv_v = wrapper.decode_step(nxt, pkv_v)
        logits_b, pkv_b = wrapper.decode_step(nxt, pkv_b)
        logits_v = logits_v[:, -1, :]
        logits_b = logits_b[:, -1, :]
    return wrapper.tokenizer.decode(gen, skip_special_tokens=True)


@torch.no_grad()
def ours_sbc_decode(wrapper: LlavaWrapper, segmenter: PanopticSegmenter,
                    image: Image.Image, question: str,
                    max_new_tokens: int = 64,
                    boost_factor: float = 1.8, top_k: int = 2, nms_iou: float = 0.5,
                    lookahead: int = 8, use_sentence_lookahead: bool = True,
                    pmi_alpha: float = 1.0, beta: float = 0.1,
                    gate_version: str = "v3",
                    tau_mid: float = 0.5, tau_lo: float = 0.25, tau_hi: float = 0.75,
                    image_margin_thresh: float = 0.5,
                    min_area_frac: float = 0.01, max_segments: int = 6,
                    return_route: bool = False):
    """``gate_version``:
      v3 (default): PMI iff (H >= tau_mid AND blank-agreement); else MSB.
      v2: H < tau_lo -> MSB; H > tau_hi & blank-agreement -> PMI; else baseline.
      lazy: PMI iff blank-agreement (and margin guard passes); else MSB.
            No H in the rule; segmentation/occlusion deferred to the MSB route.
      lazy_attn: as lazy, but MSB ranks segments by attention mass
            (one extra prefill) instead of K occlusion passes (E5).
    """
    from .baseline import greedy_decode

    def _fb():
        return greedy_decode(wrapper, image, question, max_new_tokens)

    enc = wrapper.prepare_inputs(image, question)
    input_ids = enc["input_ids"]
    pixel_v = enc["pixel_values"]
    attn = enc.get("attention_mask")

    # "lazy" gate (reframed paper, E1): the router needs no attribution.
    # Segmentation and the K occlusion passes are deferred into the MSB
    # route, so a PMI-routed (or margin-vetoed) question never pays for them.
    lazy = gate_version in ("lazy", "lazy_attn")
    attn_rank = (gate_version == "lazy_attn")

    segments = None
    if not lazy:
        segments = segmenter.segment(image, min_area_frac=min_area_frac,
                                     max_segments=max_segments)
        if len(segments) < 2:
            out = _fb()
            return (out, "fallback") if return_route else out

    span, base_lp, v_prompt_logits, v_prompt_pkv = _lookahead_with_logp(
        wrapper, input_ids, pixel_v, attn, use_sentence_lookahead, lookahead,
        max_steps=32)
    if not span:
        out = _fb()
        return (out, "fallback") if return_route else out

    phis = None
    H = None
    if not lazy:
        phis = _shap_phis(wrapper, image, segments, input_ids, pixel_v, attn,
                          span, base_lp=base_lp)
        H = _norm_entropy(phis)

    # Filled in by _blank_agrees() so _pmi() can reuse the blank prefill.
    blank_prefill: List = [None, None]   # [prompt_logits, prompt_pkv]

    def _blank_agrees():
        ok, bl, bp = _blank_span_matches(wrapper, input_ids,
                                         torch.zeros_like(pixel_v), attn, span,
                                         use_sentence_lookahead, lookahead,
                                         max_steps=32)
        blank_prefill[0], blank_prefill[1] = bl, bp
        return ok

    def _msb():
        nonlocal segments, phis, H
        if segments is None:  # lazy route: segment + attribute only here
            segments = segmenter.segment(image, min_area_frac=min_area_frac,
                                         max_segments=max_segments)
            if len(segments) < 2:
                return (_fb(), "fallback")
        if phis is None:
            if attn_rank:  # E5: rank segments by attention mass, no occlusion
                phis = _attn_segment_mass(wrapper, input_ids, pixel_v, attn,
                                          segments)
            else:
                phis = _shap_phis(wrapper, image, segments, input_ids, pixel_v,
                                  attn, span, base_lp=base_lp)
            H = _norm_entropy(phis)
        out = _generate_msb(wrapper, input_ids, pixel_v, attn, segments, phis,
                            top_k, nms_iou, boost_factor, max_new_tokens)
        return (out, "msb") if out is not None else (_fb(), "fallback")

    def _pmi():
        out = _generate_pmi(wrapper, input_ids, pixel_v, attn, pmi_alpha, beta,
                            max_new_tokens,
                            v_prefill=(v_prompt_logits, v_prompt_pkv),
                            b_prefill=(blank_prefill[0], blank_prefill[1]))
        return (out, "pmi")

    # Optional image-margin guard: if the image-conditioned top-token margin
    # already exceeds image_margin_thresh, skip PMI (which would only undo a
    # confident, image-grounded answer). Computed from v_prompt_logits once.
    def _image_margin_ok():
        if image_margin_thresh <= 0:
            return False  # guard disabled
        probs = torch.softmax(v_prompt_logits.float(), dim=-1)
        top2 = probs.topk(2, dim=-1).values[0]
        return float((top2[0] - top2[1]).item()) > image_margin_thresh

    if gate_version == "v2":
        if H < tau_lo:
            out, route = _msb()
        elif H > tau_hi and _blank_agrees():
            if _image_margin_ok():
                out, route = _fb(), "baseline(margin-guard)"
            else:
                out, route = _pmi()
        else:
            out, route = _fb(), ("baseline(blank-disagree)" if H > tau_hi
                                 else "baseline")
    elif gate_version == "pmi_guard_only":
        # Ablation: always try PMI; only the image-margin guard can veto.
        # No H or blank-agreement gating. _blank_agrees() is still called
        # for the side effect of populating blank_prefill, which _pmi() needs.
        _blank_agrees()
        if _image_margin_ok():
            out, route = _fb(), "baseline(margin-guard)"
        else:
            out, route = _pmi()
    elif gate_version in ("no_h", "lazy", "lazy_attn"):
        # no_h: App K ablation - drop the H>=tau condition; keep blank-agree
        #       + margin (attribution still computed eagerly, for logging).
        # lazy: same routing rule, but segmentation + occlusion run only if
        #       the MSB route is taken (see _msb). This is the E1 decoder.
        if _blank_agrees():
            if _image_margin_ok():
                out, route = _fb(), "baseline(margin-guard)"
            else:
                out, route = _pmi()
        else:
            out, route = _msb()
    elif gate_version == "logit_h":
        # App K ablation: replace the attribution entropy H with top-K logit entropy.
        top = v_prompt_logits[0].float().topk(100).values
        p = torch.softmax(top, dim=-1)
        H_alt = float(-(p * torch.log(p.clamp_min(1e-12))).sum().item() / math.log(100))
        if H_alt >= tau_mid and _blank_agrees():
            if _image_margin_ok():
                out, route = _fb(), "baseline(margin-guard)"
            else:
                out, route = _pmi()
        else:
            out, route = _msb()
    elif gate_version == "attn_h":
        # App K ablation: replace the attribution entropy H with last-layer attention entropy
        # aggregated to the same panoptic segments.
        out_a = wrapper.prefill(input_ids, pixel_v, attn, output_attentions=True)
        last = out_a.attentions[-1][0].mean(dim=0)[-1].float()
        vis_pos = wrapper.visual_token_positions(input_ids)
        try:
            grid = wrapper.visual_grid(input_ids)
        except TypeError:
            grid = wrapper.visual_grid()
        seg_attn = []
        for seg in segments:
            abs_idx = _segment_to_visual_token_indices(seg.mask, vis_pos, grid=grid)
            seg_attn.append(float(last[abs_idx].sum().item()) if abs_idx else 0.0)
        a = np.asarray(seg_attn, dtype=np.float64)
        if a.sum() > 0 and len(a) > 1:
            a = a / a.sum()
            H_alt = float(-(a * np.log(np.clip(a, 1e-12, 1))).sum() / math.log(len(a)))
        else:
            H_alt = 1.0
        del out_a
        if H_alt >= tau_mid and _blank_agrees():
            if _image_margin_ok():
                out, route = _fb(), "baseline(margin-guard)"
            else:
                out, route = _pmi()
        else:
            out, route = _msb()
    elif gate_version == "task_pmi":
        # App K ablation: pretend the task is over-spread for every step
        # (i.e. an oracle task-label gate for binary-QA). Keeps blank-agree
        # and the margin guard only.
        if _blank_agrees():
            if _image_margin_ok():
                out, route = _fb(), "baseline(margin-guard)"
            else:
                out, route = _pmi()
        else:
            out, route = _msb()
    elif gate_version == "task_msb":
        # App K ablation: pretend the task is over-concentration for every
        # step (oracle task-label gate for captioning).
        out, route = _msb()
    else:  # v3
        if H >= tau_mid and _blank_agrees():
            if _image_margin_ok():
                out, route = _fb(), "baseline(margin-guard)"
            else:
                out, route = _pmi()
        else:
            out, route = _msb()
    # Explicit cleanup to mitigate inter-call KV-cache accumulation (PMI route
    # keeps two pkvs: v_prefill + blank_prefill).
    del v_prompt_logits, v_prompt_pkv, blank_prefill, phis
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (out, route) if return_route else out
