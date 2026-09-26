"""Model wrapper factory.

``build_wrapper`` returns the right wrapper for a HF model id so the
benchmark scripts stay model-agnostic. All wrappers expose the same
interface (prepare_inputs / prefill / decode_step / visual_token_positions /
visual_grid / logp_spans).
"""

from __future__ import annotations

import torch


def build_wrapper(model_name: str, dtype=torch.float16,
                  attn_implementation: str = "eager",
                  load_in_8bit: bool = False,
                  device_map: str = None):
    name = model_name.lower()
    if "qwen2.5-vl" in name or "qwen2_5_vl" in name or "qwen2.5vl" in name:
        from .qwen2_5vl_wrapper import Qwen2_5VLWrapper
        return Qwen2_5VLWrapper(model_name=model_name, dtype=dtype,
                                attn_implementation=attn_implementation,
                                load_in_8bit=load_in_8bit)
    if "qwen2-vl" in name or "qwen2_vl" in name:
        from .qwen2vl_wrapper import Qwen2VLWrapper
        return Qwen2VLWrapper(model_name=model_name, dtype=dtype,
                              attn_implementation=attn_implementation,
                              load_in_8bit=load_in_8bit)
    if "llava" in name:
        from .llava_wrapper import LlavaWrapper
        kwargs = dict(model_name=model_name, dtype=dtype,
                      attn_implementation=attn_implementation,
                      load_in_8bit=load_in_8bit)
        if device_map is not None:
            kwargs["device_map"] = device_map
        return LlavaWrapper(**kwargs)
    raise ValueError(f"no wrapper registered for model id: {model_name!r}")
