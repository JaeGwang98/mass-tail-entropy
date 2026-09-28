"""LLaVA-1.5-7B wrapper used by all four decoding methods.

The wrapper deliberately stays thin: it owns the model + processor and provides
just enough plumbing (`prefill`, `decode_step`, `prepare_inputs`) for the
custom decoding loops in src/decoding/*.py to do their own per-token logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import (AutoProcessor,
                          LlavaForConditionalGeneration)

from ..utils.common import llava_prompt


@dataclass
class PrefillOutput:
    logits: torch.Tensor          # (1, expanded_seq_len, vocab)
    past_key_values: object       # transformers Cache object
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    expanded_seq_len: int = 0


class LlavaWrapper:
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf",
                 device: str = "cuda", dtype: torch.dtype = torch.float16,
                 attn_implementation: str = "eager",
                 load_in_8bit: bool = False,
                 device_map: str = None):
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_name)
        load_kwargs = dict(torch_dtype=dtype,
                           attn_implementation=attn_implementation)
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True)
            load_kwargs["device_map"] = {"": device}
        elif device_map is not None:
            # multi-GPU split for fp16 (e.g., 13B doesn't fit on single 24GB)
            load_kwargs["device_map"] = device_map
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, **load_kwargs).eval()
        if not load_in_8bit and device_map is None:
            self.model = self.model.to(device)

        # Convenience handles
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = self._image_token_id()
        # LLaVA-1.5 with CLIP-ViT-L/14-336 emits 24x24 = 576 visual tokens
        self.num_image_tokens = 576

    # ------------------------------------------------------------------
    # input preparation
    # ------------------------------------------------------------------
    def _image_token_id(self) -> int:
        cfg = getattr(self.model.config, "image_token_index", None) \
              or getattr(self.model.config, "image_token_id", None)
        if cfg is not None:
            return int(cfg)
        return int(self.tokenizer.convert_tokens_to_ids("<image>"))

    def prepare_inputs(self, image: Image.Image, question: str):
        """Returns ``input_ids`` (with the <image> placeholder) and the
        preprocessed ``pixel_values``.  The placeholder is expanded to 576
        visual tokens internally by the model on forward."""
        prompt = llava_prompt(question)
        enc = self.processor(images=image, text=prompt, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        if "pixel_values" in enc:
            enc["pixel_values"] = enc["pixel_values"].to(self.dtype)
        return enc

    # ------------------------------------------------------------------
    # Visual-token bookkeeping.  Newer LLaVA HF integration expands the
    # single <image> id into ``num_image_tokens`` consecutive ids *before*
    # forward, so we can locate them by id-equality.
    # ------------------------------------------------------------------
    def expand_image_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """If the prompt only carries one <image> id, expand it to
        ``num_image_tokens`` copies so prefill positions line up with
        what the model actually consumes (HF post-2024 LLaVA convention)."""
        img_id = self.image_token_id
        flat = input_ids[0].tolist()
        if flat.count(img_id) == 1:
            idx = flat.index(img_id)
            expanded = flat[:idx] + [img_id] * self.num_image_tokens + flat[idx + 1:]
            return torch.tensor([expanded], dtype=input_ids.dtype,
                                device=input_ids.device)
        return input_ids

    def visual_token_positions(self, input_ids: torch.Tensor) -> List[int]:
        img_id = self.image_token_id
        return (input_ids[0] == img_id).nonzero(as_tuple=True)[0].tolist()

    # ------------------------------------------------------------------
    # raw forward helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, pixel_values: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False, **_ignore) -> PrefillOutput:
        # ``_ignore`` absorbs model-specific extras (image_grid_thw,
        # mm_token_type_ids) that model-agnostic decoders thread via
        # _extra_from_enc(enc); LLaVA has none, so they are simply dropped.
        out = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=output_attentions,
            return_dict=True,
        )
        return PrefillOutput(
            logits=out.logits,
            past_key_values=out.past_key_values,
            attentions=out.attentions if output_attentions else None,
            expanded_seq_len=out.logits.shape[1],
        )

    @torch.no_grad()
    def decode_step(self, last_token: torch.Tensor, past_key_values,
                    attention_mask: Optional[torch.Tensor] = None):
        """One-token step that consumes ``past_key_values`` from a previous
        prefill or decode_step."""
        out = self.model(
            input_ids=last_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return out.logits, out.past_key_values

    # ------------------------------------------------------------------
    # Model-agnostic interface (also implemented by Qwen2VLWrapper)
    # ------------------------------------------------------------------
    def visual_grid(self, input_ids: Optional[torch.Tensor] = None
                    ) -> Tuple[int, int]:
        """(grid_h, grid_w) of the visual-token grid. LLaVA-1.5 is a fixed
        24x24 = 576 grid regardless of image. ``input_ids`` is accepted for
        signature parity with dynamic-grid models (Qwen2-VL)."""
        return (24, 24)

    @torch.no_grad()
    def logp_spans(self, images: List[Image.Image], prompt_ids: torch.Tensor,
                   span: List[int], max_batch: int = 8) -> torch.Tensor:
        """Teacher-forced log p(span | prompt, image) for each image in
        ``images``. Returns a (len(images),) tensor.

        LLaVA's image-token count is fixed (576) and content-independent, so
        the same ``prompt_ids`` are reused for every image and only
        ``pixel_values`` is swapped — the K images are stacked on the batch
        axis and run in chunked forwards (the original batched-occlusion path)."""
        if not span:
            return torch.zeros(len(images), device=self.device)
        pix_list = []
        for im in images:
            enc = self.processor.image_processor(images=im,
                                                 return_tensors="pt")
            pix_list.append(enc["pixel_values"].to(self.device,
                                                   dtype=self.dtype))
        pixel_batch = torch.cat(pix_list, dim=0)              # (N,3,H,W)

        span_t = torch.tensor([list(span)], device=prompt_ids.device,
                              dtype=prompt_ids.dtype)
        full_ids = torch.cat([prompt_ids, span_t], dim=1)     # (1,S)
        L = len(span)
        span_idx = torch.tensor(span, device=prompt_ids.device,
                                dtype=torch.long)
        # Pre-refactor parity: the old _shap_phis_batched passed an explicit
        # all-ones attention_mask (LLaVA's prepare_inputs returns one) to the
        # occlusion forward. Replicate it so fp16 reduction order — and hence
        # phi, the H gate, and top-K segment selection — stays bit-identical.
        attn_ones = torch.ones_like(full_ids)
        lps = []
        N = pixel_batch.shape[0]
        for start in range(0, N, max_batch):
            chunk = pixel_batch[start:start + min(max_batch, N - start)]
            b = chunk.shape[0]
            out = self.model(input_ids=full_ids.repeat(b, 1),
                             pixel_values=chunk,
                             attention_mask=attn_ones.repeat(b, 1),
                             use_cache=False, return_dict=True)
            logits = out.logits.float()                       # (b,S,V)
            span_logits = logits[:, -L - 1:-1, :]
            log_probs = torch.log_softmax(span_logits, dim=-1)
            lp = log_probs[:, torch.arange(L, device=log_probs.device),
                           span_idx].sum(dim=1)               # (b,)
            lps.append(lp)
            del out, logits
        return torch.cat(lps, dim=0)                           # (N,)
