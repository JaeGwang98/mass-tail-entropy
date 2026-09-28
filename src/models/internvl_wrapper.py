"""InternVL3 wrapper (HF-native ``OpenGVLab/InternVL3-*-hf`` checkpoints)
exposing the subset of the ``LlavaWrapper`` interface that the H diagnostic
(``scripts/h_distribution_anymodel.py``) needs: ``prepare_inputs`` /
``prefill`` / ``decode_step`` / ``logp_spans`` / ``tokenizer``.

InternVL3 is the third architecture family of the paper's breadth check: its
vision encoder is InternViT (the LLaVA and Qwen backbones use their own ViTs).

Image handling: InternVL's processor tiles large images into up to 12
448x448 crops plus a thumbnail by default. We disable tiling
(``crop_to_patches=False``) so every image -- original, occluded or blank --
maps to the same single 448x448 tile and the same 256 ``<IMG_CONTEXT>``
tokens. Occlusion is applied to the original-resolution image before
preprocessing, exactly as for LLaVA / Qwen, so the leave-one-segment-out
protocol is unchanged; only the model-side resolution differs (like LLaVA-1.5,
which also sees a single fixed-size square view).

The MSB actuator (visual-token boosting) is not implemented for this family:
the paper evaluates the decoder on the two 7B backbones only, and the
diagnostic uses forward passes alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, InternVLForConditionalGeneration


@dataclass
class PrefillOutput:
    logits: torch.Tensor
    past_key_values: object
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    expanded_seq_len: int = 0


class InternVLWrapper:
    def __init__(self, model_name: str = "OpenGVLab/InternVL3-8B-hf",
                 device: Optional[str] = None,
                 dtype: torch.dtype = torch.bfloat16,
                 attn_implementation: str = "eager",
                 load_in_8bit: bool = False):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # InternVL3's language model is Qwen2.5, which is unstable in float16
        # (same reason the Qwen wrappers coerce); the official dtype is bf16.
        if dtype == torch.float16 and device != "cpu":
            dtype = torch.bfloat16
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_name)
        load_kwargs = dict(dtype=dtype, attn_implementation=attn_implementation)
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True)
            load_kwargs["device_map"] = {"": device}
        self.model = InternVLForConditionalGeneration.from_pretrained(
            model_name, **load_kwargs).eval()
        if not load_in_8bit:
            self.model = self.model.to(device)
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = int(self.tokenizer.convert_tokens_to_ids(
            self.processor.image_token))

    # ------------------------------------------------------------------
    # input preparation
    # ------------------------------------------------------------------
    def _prompt(self, question: str) -> str:
        msgs = [{"role": "user",
                 "content": [{"type": "image"},
                             {"type": "text", "text": question}]}]
        return self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False)

    def _pixels(self, image: Image.Image) -> torch.Tensor:
        enc = self.processor.image_processor(
            images=image, crop_to_patches=False, return_tensors="pt")
        return enc["pixel_values"].to(self.device, dtype=self.dtype)

    def prepare_inputs(self, image: Image.Image, question: str):
        enc = self.processor(images=image, text=self._prompt(question),
                             crop_to_patches=False, return_tensors="pt")
        out = {k: (v.to(self.device) if torch.is_tensor(v) else v)
               for k, v in enc.items()}
        out["pixel_values"] = out["pixel_values"].to(self.dtype)
        return out

    def visual_token_positions(self, input_ids: torch.Tensor) -> List[int]:
        return (input_ids[0] == self.image_token_id).nonzero(
            as_tuple=True)[0].tolist()

    def visual_grid(self, input_ids: Optional[torch.Tensor] = None):
        raise NotImplementedError(
            "MSB (visual-token boosting) is not implemented for InternVL; "
            "this wrapper supports the H diagnostic only.")

    # ------------------------------------------------------------------
    # raw forward helpers (positional signature == LlavaWrapper)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, pixel_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False, **_ignore) -> PrefillOutput:
        if attention_mask is not None and attention_mask.dim() == 4:
            raise NotImplementedError(
                "4-D boost masks (MSB) are not supported for InternVL.")
        out = self.model(input_ids=input_ids, pixel_values=pixel_values,
                         attention_mask=attention_mask, use_cache=True,
                         output_attentions=output_attentions,
                         return_dict=True)
        return PrefillOutput(
            logits=out.logits,
            past_key_values=out.past_key_values,
            attentions=out.attentions if output_attentions else None,
            expanded_seq_len=out.logits.shape[1],
        )

    @torch.no_grad()
    def decode_step(self, last_token: torch.Tensor, past_key_values,
                    attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is not None and attention_mask.dim() == 4:
            raise NotImplementedError(
                "4-D boost masks (MSB) are not supported for InternVL.")
        out = self.model(input_ids=last_token, past_key_values=past_key_values,
                         attention_mask=attention_mask, use_cache=True,
                         return_dict=True)
        return out.logits, out.past_key_values

    # ------------------------------------------------------------------
    # occlusion scoring
    # ------------------------------------------------------------------
    @torch.no_grad()
    def logp_spans(self, images: List[Image.Image], prompt_ids: torch.Tensor,
                   span: List[int], max_batch: int = 8) -> torch.Tensor:
        """Teacher-forced log p(span | prompt, image) per image. Every image
        maps to one 448x448 tile, so the prompt (with its 256 image tokens) is
        shared and images are batched on the batch axis like LLaVA."""
        if not span:
            return torch.zeros(len(images), device=self.device)
        span_t = torch.tensor([list(span)], device=prompt_ids.device,
                              dtype=prompt_ids.dtype)
        full_ids = torch.cat([prompt_ids, span_t], dim=1)          # (1,S)
        L = len(span)
        span_idx = torch.tensor(span, device=prompt_ids.device,
                                dtype=torch.long)
        lps = []
        for i in range(0, len(images), max_batch):
            chunk = images[i:i + max_batch]
            pix = torch.cat([self._pixels(im) for im in chunk], dim=0)
            ids = full_ids.expand(len(chunk), -1)
            out = self.model(input_ids=ids, pixel_values=pix,
                             use_cache=False, return_dict=True)
            span_logits = out.logits[:, -L - 1:-1, :].float()
            log_probs = torch.log_softmax(span_logits, dim=-1)
            lp = log_probs[:, torch.arange(L, device=log_probs.device),
                           span_idx].sum(dim=-1)
            lps.append(lp)
            del out, span_logits, log_probs
        return torch.cat(lps, dim=0)                                # (N,)
