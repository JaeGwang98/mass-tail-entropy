"""Forward-diffusion image distortion used by VCD (Eq. 2 in Leng et al., CVPR 2024).

Mirrors the official VCD implementation
(https://github.com/DAMO-NLP-SG/VCD/blob/main/experiments/eval/vcd_utils/vcd_add_noise.py).
"""

from __future__ import annotations

import torch


def add_diffusion_noise(image_tensor: torch.Tensor, noise_step: int,
                        num_steps: int = 1000) -> torch.Tensor:
    """Apply DDPM-style forward diffusion noise to a pixel-value tensor.

    Args:
        image_tensor: preprocessed pixel values, e.g. shape (C, H, W) or (B, C, H, W).
        noise_step:   t in [0, num_steps).  Larger -> more degraded.
        num_steps:    total diffusion horizon.  VCD uses 1000.
    """
    if noise_step < 0 or noise_step >= num_steps:
        raise ValueError(f"noise_step={noise_step} out of [0, {num_steps})")
    betas = torch.linspace(-6, 6, num_steps, device=image_tensor.device)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1.0 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)

    sqrt_alpha = alphas_prod[noise_step].sqrt().to(image_tensor.dtype)
    sqrt_one_minus = (1.0 - alphas_prod[noise_step]).sqrt().to(image_tensor.dtype)
    noise = torch.randn_like(image_tensor)
    return sqrt_alpha * image_tensor + sqrt_one_minus * noise
