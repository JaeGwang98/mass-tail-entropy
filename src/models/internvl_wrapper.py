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

MSB actuator (added for the camera-ready third-family decoding run):
``visual_grid`` returns the LLM-side token grid (448/14 = 32x32 InternViT
patches -> pixel-shuffle x0.5 -> 16x16 = 256 tokens, flattened row-major, see
``InternVLModel.pixel_shuffle``), and ``prefill`` / ``decode_step`` accept the
4-D additive boost masks built by ``ours_v4._build_boost_mask`` /
``_build_step_boost_mask``. InternVL's language model is Qwen2 with 1-D RoPE
positions derived from ``cache_position`` (not from the mask), and
``transformers.masking_utils`` passes a 4-D mask through unchanged, so -- unlike
Qwen2.5-VL's M-RoPE -- no explicit position ids are needed: the same mask is
added to the pre-softmax scores of every layer / head (eager attention).
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
        cfg = self.model.config
        vc = cfg.vision_config
        img = vc.image_size[0] if isinstance(vc.image_size, (list, tuple)) \
            else vc.image_size
        pch = vc.patch_size[0] if isinstance(vc.patch_size, (list, tuple)) \
            else vc.patch_size
        side = int(round(int(img) // int(pch) * float(cfg.downsample_ratio)))
        self.grid_h = self.grid_w = side                     # 16 for 448/14

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

    @property
    def num_image_tokens(self) -> int:
        return self.grid_h * self.grid_w

    def expand_image_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The processor already expands ``<IMG_CONTEXT>``; no-op (parity)."""
        return input_ids

    def visual_grid(self, input_ids: Optional[torch.Tensor] = None
                    ) -> Tuple[int, int]:
        """(grid_h, grid_w) of the LLM-side visual-token grid. Tiling is
        disabled, so every image is one 448x448 tile -> fixed 16x16 grid,
        row-major (pixel_shuffle keeps (row, col) order)."""
        if input_ids is not None:
            n = len(self.visual_token_positions(input_ids))
            if n != self.grid_h * self.grid_w:
                raise RuntimeError(
                    f"expected {self.grid_h * self.grid_w} image tokens, "
                    f"got {n} (tiling must stay disabled)")
        return (self.grid_h, self.grid_w)

    # ------------------------------------------------------------------
    # raw forward helpers (positional signature == LlavaWrapper)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, pixel_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False, **_ignore) -> PrefillOutput:
        # A 4-D additive mask (MSB boost) is forwarded as-is: masking_utils
        # returns 4-D masks unchanged and eager attention adds it to the
        # scores of every layer / head. Cast to the model dtype for safety.
        if attention_mask is not None and attention_mask.dim() == 4:
            attention_mask = attention_mask.to(self.dtype)
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
        # A 4-D additive mask (MSB boost) is forwarded as-is: masking_utils
        # returns 4-D masks unchanged and eager attention adds it to the
        # scores of every layer / head. Cast to the model dtype for safety.
        if attention_mask is not None and attention_mask.dim() == 4:
            attention_mask = attention_mask.to(self.dtype)
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
