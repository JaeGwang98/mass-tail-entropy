"""Paying Attention to Image Tokens (PAI, Chen et al., 2025).

PAI boosts attention to visual tokens at decode time by modifying the raw
attention logits (pre-softmax) in a chosen range of LLaMA/Qwen2.5-VL layers:

    attn[..., -1, img_start:img_end] += attn[..., -1, img_start:img_end].abs() * alpha

Only the last query row (the newly generated token) is modified; prefill rows
are left untouched.  This encourages the model to keep attending to image
content during autoregressive generation.

Optional CFG (Classifier-Free Guidance): a second forward pass is run with a
*text-only* negative prompt (the image tokens removed) and logits are blended:

    logits = (1 + gamma) * logits_full - gamma * logits_text_only

Supported wrappers:
- LlavaWrapper  : uses LlamaAttention path (wrapper.model.language_model.model.layers)
- Qwen2_5VLWrapper: uses Qwen2_5_VLAttention path (wrapper.model.model.layers)
  CFG is disabled for Qwen2.5-VL (text-only prefill requires pixel_values=None which
  triggers a different code path in the Qwen2.5-VL model; use_cfg is forced False).

Reference: attention.py / chair_eval.py from the PAI repo.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import List, Optional, Union

import torch
from PIL import Image

from ..models.llava_wrapper import LlavaWrapper


# ---------------------------------------------------------------------------
# Patched forward for LLaMA self-attention layers (LLaVA path)
# ---------------------------------------------------------------------------
def _pai_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    """Drop-in replacement for LlamaAttention.forward with PAI modification."""
    import math
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Reshape to (bsz, num_heads, seq, head_dim)
    head_dim = self.head_dim
    num_heads = self.num_heads
    num_kv_heads = getattr(self, "num_key_value_heads", num_heads)

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states   = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    # Rotary embeddings
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if hasattr(past_key_value, "get_usable_length"):
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value[0].shape[-2]

    if position_embeddings is not None:
        cos, sin = position_embeddings
    else:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # KV cache update
    if past_key_value is not None:
        if hasattr(past_key_value, "update"):
            cache_kwargs_inner = {"sin": sin, "cos": cos}
            if cache_position is not None:
                cache_kwargs_inner["cache_position"] = cache_position
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs_inner
            )
        else:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

    # GQA repeat if needed
    if num_kv_heads != num_heads:
        n_rep = num_heads // num_kv_heads
        key_states   = key_states.repeat_interleave(n_rep, dim=1)
        value_states = value_states.repeat_interleave(n_rep, dim=1)

    kv_seq_len = key_states.shape[-2]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # ---- PAI modification (pre-softmax) ------------------------------------
    if getattr(self, "_pai_use_attn", False) and not getattr(self, "_pai_use_cfg", False):
        s = self._pai_img_start
        e = self._pai_img_end
        alpha = self._pai_alpha
        # Only modify last query row (decode step token)
        attn_weights[:, :, -1, s:e] = (
            attn_weights[:, :, -1, s:e].abs() * alpha
            + attn_weights[:, :, -1, s:e]
        )
    # ---- end PAI -----------------------------------------------------------

    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    past_kv_out = past_key_value if hasattr(past_key_value, "update") else None
    return attn_output, attn_weights, past_kv_out


# ---------------------------------------------------------------------------
# Patched forward for Qwen2.5-VL self-attention layers
# ---------------------------------------------------------------------------
def _pai_qwen25_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    """Drop-in replacement for Qwen2_5_VLAttention.forward with PAI modification.

    Replaces the attention_interface dispatch with an inline eager computation
    so the PAI boost can be injected between mask-add and softmax.
    M-RoPE is applied via apply_multimodal_rotary_pos_emb (position_embeddings
    always supplied by the Qwen2.5-VL model's forward pass).
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb,
    )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states   = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    # M-RoPE: position_embeddings is always (cos, sin) passed from the layer norm
    # stack above (Qwen2.5-VL never falls back to a local rotary_emb call).
    if position_embeddings is None:
        raise RuntimeError(
            "_pai_qwen25_attn_forward: position_embeddings is None. "
            "Qwen2.5-VL always provides this; something is wrong upstream."
        )
    cos, sin = position_embeddings
    mrope_section = self.config.rope_parameters["mrope_section"]
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, mrope_section
    )

    # KV cache update (DynamicCache API used by Qwen2.5-VL)
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    # GQA expand keys/values to match query head count
    num_kv_groups = self.num_key_value_groups
    if num_kv_groups > 1:
        key_states   = key_states.repeat_interleave(num_kv_groups, dim=1)
        value_states = value_states.repeat_interleave(num_kv_groups, dim=1)

    # Scaled dot-product attention weights (pre-softmax)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # ---- PAI modification (pre-softmax) ------------------------------------
    if getattr(self, "_pai_use_attn", False):
        s = self._pai_img_start
        e = self._pai_img_end
        alpha = self._pai_alpha
        # Only modify last query row (decode step token)
        attn_weights[:, :, -1, s:e] = (
            attn_weights[:, :, -1, s:e].abs() * alpha
            + attn_weights[:, :, -1, s:e]
        )
    # ---- end PAI -----------------------------------------------------------

    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query_states.dtype)
    attn_weights = torch.nn.functional.dropout(
        attn_weights, p=0.0, training=self.training
    )

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Context manager: wrapper-aware patch / unpatch layers
# ---------------------------------------------------------------------------
@contextmanager
def _pai_patch(wrapper, img_start: int, img_end: int,
               alpha: float, start_layer: int, end_layer: int,
               use_attn: bool = True, use_cfg: bool = False):
    """Monkey-patch self-attention layers in-place for PAI, then restore.

    Automatically detects wrapper type to select the correct layer path and
    patched forward function:
    - LLaVA  : wrapper.model.language_model.model.layers  -> _pai_attn_forward
    - Qwen2.5-VL: wrapper.model.model.layers              -> _pai_qwen25_attn_forward
    """
    # Detect wrapper type and select path (transformers 5.x compatible).
    m = wrapper.model
    # HF 5.x LLaVA: model.model.language_model.layers
    if (hasattr(m, "model") and hasattr(m.model, "language_model")
            and hasattr(m.model.language_model, "layers")):
        layers = m.model.language_model.layers
        patched_forward = _pai_attn_forward
    # HF 4.x LLaVA legacy: model.language_model.model.layers
    elif (hasattr(m, "language_model") and hasattr(m.language_model, "model")
            and hasattr(m.language_model.model, "layers")):
        layers = m.language_model.model.layers
        patched_forward = _pai_attn_forward
    # Qwen2.5-VL: model.model.layers
    elif hasattr(m, "model") and hasattr(m.model, "layers"):
        layers = m.model.layers
        patched_forward = _pai_qwen25_attn_forward
    else:
        raise RuntimeError(
            "_pai_patch: cannot determine layer path for wrapper type "
            f"{type(m).__name__}"
        )

    saved: List[tuple] = []
    for i in range(start_layer, min(end_layer, len(layers))):
        attn = layers[i].self_attn
        saved.append((i, attn.forward))
        attn._pai_use_attn  = use_attn
        attn._pai_use_cfg   = use_cfg
        attn._pai_alpha     = alpha
        attn._pai_img_start = img_start
        attn._pai_img_end   = img_end
        attn.forward = types.MethodType(patched_forward, attn)

    try:
        yield
    finally:
        for i, orig_fwd in saved:
            attn = layers[i].self_attn
            attn.forward = orig_fwd
            for attr in ("_pai_use_attn", "_pai_use_cfg", "_pai_alpha",
                         "_pai_img_start", "_pai_img_end"):
                attn.__dict__.pop(attr, None)


# ---------------------------------------------------------------------------
# Text-only negative prompt for CFG (LLaVA only)
# ---------------------------------------------------------------------------
def _build_text_only_ids(wrapper, input_ids: torch.Tensor) -> torch.Tensor:
    """Remove image tokens from input_ids to form the CFG negative prompt."""
    img_id = wrapper.image_token_id
    flat = input_ids[0].tolist()
    text_only = [t for t in flat if t != img_id]
    return torch.tensor([text_only], dtype=input_ids.dtype, device=input_ids.device)


def _is_qwen25vl(wrapper) -> bool:
    """True when wrapper is a Qwen2_5VLWrapper."""
    # Avoid a hard import to keep the file model-agnostic
    return type(wrapper).__name__ == "Qwen2_5VLWrapper"


# ---------------------------------------------------------------------------
# Top-level decoding entry point
# ---------------------------------------------------------------------------
@torch.no_grad()
def pai_decode(
    wrapper,
    image: Image.Image,
    question: str,
    max_new_tokens: int = 512,
    alpha: float = 0.5,
    gamma_cfg: float = 1.1,
    use_cfg: bool = True,
    start_layer: int = 2,
    end_layer: int = 32,
) -> str:
    """PAI decoding with optional CFG.

    Parameters
    ----------
    wrapper        : LlavaWrapper or Qwen2_5VLWrapper instance
    image          : PIL image
    question       : text question / prompt
    max_new_tokens : generation budget
    alpha          : PAI attention amplification strength
    gamma_cfg      : CFG guidance scale (only used when use_cfg=True and not Qwen2.5-VL)
    use_cfg        : whether to run the text-only CFG branch.
                     Forced False for Qwen2.5-VL (pixel_values=None path unsupported).
    start_layer    : first layer to patch (inclusive)
    end_layer      : last layer to patch (exclusive)
    """
    # CFG is not supported for Qwen2.5-VL (text-only prefill differs structurally)
    if _is_qwen25vl(wrapper) and use_cfg:
        use_cfg = False

    enc = wrapper.prepare_inputs(image, question)
    input_ids    = enc["input_ids"]
    pixel_values = enc["pixel_values"]
    attn_mask    = enc.get("attention_mask")

    # Locate image token span in the (possibly single-<image>) input_ids
    # visual_token_positions returns the indices in the pre-expanded ids;
    # the model expands them internally, so we use the expanded form.
    expanded_ids = wrapper.expand_image_tokens(input_ids)
    vis_pos = wrapper.visual_token_positions(expanded_ids)
    if not vis_pos:
        # Fallback: no image tokens found — run greedy without patching
        from .baseline import greedy_decode
        return greedy_decode(wrapper, image, question,
                             max_new_tokens=max_new_tokens)

    img_start = vis_pos[0]
    img_end   = vis_pos[-1] + 1   # exclusive

    eos = int(wrapper.tokenizer.eos_token_id)

    # --- CFG: text-only negative prefill (LLaVA only) ----------------------
    pkv_neg: Optional[object] = None
    logits_neg: Optional[torch.Tensor] = None
    if use_cfg:
        neg_ids = _build_text_only_ids(wrapper, expanded_ids)
        # Text-only forward: no pixel_values, no PAI patch
        out_neg = wrapper.prefill(neg_ids, pixel_values=None,
                                  attention_mask=None)
        logits_neg = out_neg.logits[:, -1, :]
        pkv_neg    = out_neg.past_key_values

    # --- Full-image prefill with PAI patch ---------------------------------
    with _pai_patch(wrapper, img_start, img_end, alpha,
                    start_layer, end_layer,
                    use_attn=True, use_cfg=False):
        out_full = wrapper.prefill(input_ids, pixel_values, attn_mask)

    logits_full = out_full.logits[:, -1, :]
    pkv_full    = out_full.past_key_values

    # --- Decode loop -------------------------------------------------------
    generated: List[int] = []
    for _ in range(max_new_tokens):
        if use_cfg and logits_neg is not None:
            # CFG blend: (1 + gamma) * logits_full - gamma * logits_neg
            logits = (1.0 + gamma_cfg) * logits_full - gamma_cfg * logits_neg
        else:
            logits = logits_full

        nxt = logits.argmax(-1, keepdim=True)
        token_id = int(nxt.item())
        if token_id == eos:
            break
        generated.append(token_id)

        # Full-image decode step with PAI patch
        with _pai_patch(wrapper, img_start, img_end, alpha,
                        start_layer, end_layer,
                        use_attn=True, use_cfg=False):
            logits_full, pkv_full = wrapper.decode_step(nxt, pkv_full)
        logits_full = logits_full[:, -1, :]

        # Text-only CFG step (no patch needed)
        if use_cfg:
            logits_neg, pkv_neg = wrapper.decode_step(nxt, pkv_neg)
            logits_neg = logits_neg[:, -1, :]

    return wrapper.tokenizer.decode(generated, skip_special_tokens=True)
