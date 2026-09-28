"""Qwen2.5-VL-7B wrapper exposing the same interface as ``LlavaWrapper`` so the
LOO-attribution / MSB / PMI decoders run unchanged.

Architecture is identical to Qwen2-VL (dynamic resolution, M-RoPE, same patch
merging) with the following differences handled here:

- HF class: ``Qwen2_5_VLForConditionalGeneration`` (model_type=qwen2_5_vl).
- ``get_rope_index`` on ``Qwen2_5_VLModel`` requires ``mm_token_type_ids`` as
  a positional (non-optional) argument; we always supply it.
- The statefulness strategy is identical: ``prepare_inputs`` stashes multimodal
  extras; ``prefill`` / ``logp_spans`` re-inject them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def qwen2_5vl_prompt(question: str) -> str:
    """Qwen2.5-VL chat template: system + one image + question."""
    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n<|im_start|>assistant\n")


@dataclass
class PrefillOutput:
    logits: torch.Tensor
    past_key_values: object
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    expanded_seq_len: int = 0


class Qwen2_5VLWrapper:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 attn_implementation: str = "eager",
                 load_in_8bit: bool = False):
        self.device = device
        # Qwen2.5-VL is numerically unstable in float16; coerce to bfloat16.
        if dtype == torch.float16:
            dtype = torch.bfloat16
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_name)
        load_kwargs = dict(torch_dtype=dtype,
                           attn_implementation=attn_implementation)
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True)
            load_kwargs["device_map"] = {"": device}
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, **load_kwargs).eval()
        if not load_in_8bit:
            self.model = self.model.to(device)

        self.tokenizer = self.processor.tokenizer
        self.image_token_id = int(self.tokenizer.convert_tokens_to_ids(
            "<|image_pad|>"))
        vc = self.model.config.vision_config
        self.spatial_merge_size = int(getattr(vc, "spatial_merge_size", 2))
        # Stashed per-image multimodal extras (set by prepare_inputs,
        # consumed by prefill / logp_spans).
        self._mm: dict = {}

    # ------------------------------------------------------------------
    # input preparation
    # ------------------------------------------------------------------
    def prepare_inputs(self, image: Image.Image, question: str):
        prompt = qwen2_5vl_prompt(question)
        enc = self.processor(images=image, text=prompt, return_tensors="pt")
        out = {}
        for k, v in enc.items():
            out[k] = v.to(self.device) if torch.is_tensor(v) else v
        if "pixel_values" in out:
            out["pixel_values"] = out["pixel_values"].to(self.dtype)
        # stash multimodal extras for the stateful prefill
        self._mm = {
            "image_grid_thw": out.get("image_grid_thw"),
            "mm_token_type_ids": out.get("mm_token_type_ids"),
        }
        return out

    def visual_token_positions(self, input_ids: torch.Tensor) -> List[int]:
        return (input_ids[0] == self.image_token_id).nonzero(
            as_tuple=True)[0].tolist()

    @property
    def num_image_tokens(self) -> int:
        """Dynamic in Qwen2.5-VL; callers must use visual_token_positions()."""
        return 0

    def expand_image_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The processor already expands ``<|image_pad|>`` per image_grid_thw,
        so this is a no-op (kept for interface parity with LlavaWrapper)."""
        return input_ids

    def visual_grid(self, input_ids: Optional[torch.Tensor] = None
                    ) -> Tuple[int, int]:
        """(grid_h, grid_w) of the *LLM-side* visual-token grid for the most
        recently prepared image: patch grid // spatial_merge_size."""
        thw = self._mm.get("image_grid_thw")
        if thw is None:
            raise RuntimeError("visual_grid() called before prepare_inputs()")
        t, h, w = [int(x) for x in thw[0].tolist()]
        m = self.spatial_merge_size
        return (h // m, w // m)

    # ------------------------------------------------------------------
    # raw forward helpers (positional signature == LlavaWrapper)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, pixel_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False,
                image_grid_thw: Optional[torch.Tensor] = None,
                mm_token_type_ids: Optional[torch.Tensor] = None,
                **_ignore) -> PrefillOutput:
        """Multimodal extras may be passed explicitly or fall back to the
        per-image stash from ``prepare_inputs``."""
        kwargs = dict(input_ids=input_ids,
                      pixel_values=pixel_values,
                      attention_mask=attention_mask,
                      use_cache=True,
                      output_attentions=output_attentions,
                      return_dict=True)
        thw = image_grid_thw if image_grid_thw is not None \
            else self._mm.get("image_grid_thw")
        mm = mm_token_type_ids if mm_token_type_ids is not None \
            else self._mm.get("mm_token_type_ids")
        if thw is not None:
            kwargs["image_grid_thw"] = thw
        if mm is not None:
            # mm_token_type_ids must match input_ids length; the stash covers
            # the prompt. If a span was appended (lookahead/occlusion), pad type 0.
            if mm.shape[1] != input_ids.shape[1]:
                pad = torch.zeros(mm.shape[0],
                                  input_ids.shape[1] - mm.shape[1],
                                  dtype=mm.dtype, device=mm.device)
                mm = torch.cat([mm, pad], dim=1)
            kwargs["mm_token_type_ids"] = mm

        # MSB passes a 4-D additive boost mask as ``attention_mask``.
        # Qwen2.5-VL's get_rope_index assumes a 2-D padding mask and crashes
        # on 4-D. Fix: precompute M-RoPE position ids from a clean mask and
        # pass them explicitly so forward skips get_rope_index, while the 4-D
        # mask still reaches the attention layers.
        # NOTE: Qwen2.5-VL's get_rope_index requires mm_token_type_ids as a
        # positional (non-optional) argument; always provide a fallback.
        if attention_mask is not None and attention_mask.dim() == 4:
            inner = self.model.model            # Qwen2_5_VLModel
            _mm_for_rope = kwargs.get("mm_token_type_ids")
            if _mm_for_rope is None:
                _mm_for_rope = torch.zeros(
                    input_ids.shape, dtype=torch.int, device=input_ids.device)
            pos_ids, rope_deltas = inner.get_rope_index(
                input_ids,
                mm_token_type_ids=_mm_for_rope,
                image_grid_thw=kwargs.get("image_grid_thw"),
                attention_mask=None)
            inner.rope_deltas = rope_deltas
            kwargs["position_ids"] = pos_ids
        out = self.model(**kwargs)
        return PrefillOutput(
            logits=out.logits,
            past_key_values=out.past_key_values,
            attentions=out.attentions if output_attentions else None,
            expanded_seq_len=out.logits.shape[1],
        )

    @torch.no_grad()
    def decode_step(self, last_token: torch.Tensor, past_key_values,
                    attention_mask: Optional[torch.Tensor] = None):
        """One-token step. MSB passes a 4-D per-step boost mask; fix mirrors
        Qwen2-VL wrapper: compute incremental 3-D position ids explicitly so
        the model skips its mask-dependent position path."""
        kwargs = dict(input_ids=last_token,
                      attention_mask=attention_mask,
                      past_key_values=past_key_values,
                      use_cache=True, return_dict=True)
        if attention_mask is not None and attention_mask.dim() == 4:
            inner = self.model.model
            past_len = past_key_values.get_seq_length()
            bsz, seq_len = last_token.shape[0], last_token.shape[1]
            pos = torch.arange(past_len, past_len + seq_len,
                               device=last_token.device)
            pos = pos.view(1, 1, -1).expand(3, bsz, -1)
            delta = inner.rope_deltas
            delta = delta.repeat_interleave(
                bsz // delta.shape[0], dim=0).to(pos.device)
            kwargs["position_ids"] = pos + delta
        out = self.model(**kwargs)
        return out.logits, out.past_key_values

    # ------------------------------------------------------------------
    # model-agnostic occlusion scoring (loop; can't batch flat-patch tensor)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def logp_spans(self, images: List[Image.Image], prompt_ids: torch.Tensor,
                   span: List[int], max_batch: int = 8) -> torch.Tensor:
        """Teacher-forced log p(span | prompt, image) per image.

        All occlusion variants share the original image's resolution, hence
        the same image_grid_thw and the same number of <|image_pad|> tokens,
        so ``prompt_ids`` (already expanded) is reused and only the patch
        ``pixel_values`` + ``image_grid_thw`` are swapped per image.
        """
        if not span:
            return torch.zeros(len(images), device=self.device)
        span_t = torch.tensor([list(span)], device=prompt_ids.device,
                              dtype=prompt_ids.dtype)
        full_ids = torch.cat([prompt_ids, span_t], dim=1)       # (1,S)
        L = len(span)
        span_idx = torch.tensor(span, device=prompt_ids.device,
                                dtype=torch.long)
        base_mm = self._mm.get("mm_token_type_ids")
        if base_mm is not None:
            pad = torch.zeros(base_mm.shape[0], L, dtype=base_mm.dtype,
                              device=base_mm.device)
            full_mm = torch.cat([base_mm, pad], dim=1)
        else:
            full_mm = None

        lps = []
        for im in images:
            enc = self.processor.image_processor(images=im,
                                                 return_tensors="pt")
            pix = enc["pixel_values"].to(self.device, dtype=self.dtype)
            thw = enc["image_grid_thw"].to(self.device)
            kw = dict(input_ids=full_ids, pixel_values=pix,
                      image_grid_thw=thw, use_cache=False, return_dict=True)
            if full_mm is not None:
                kw["mm_token_type_ids"] = full_mm
            out = self.model(**kw)
            logits = out.logits.float()                          # (1,S,V)
            span_logits = logits[:, -L - 1:-1, :]
            log_probs = torch.log_softmax(span_logits, dim=-1)
            lp = log_probs[0, torch.arange(L, device=log_probs.device),
                           span_idx].sum()
            lps.append(lp)
            del out, logits
        return torch.stack(lps, dim=0)                            # (N,)
