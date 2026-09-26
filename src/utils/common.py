"""Misc helpers: config loading, prompt formatting, deterministic RNG."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def llava_prompt(question: str) -> str:
    """LLaVA-1.5 (Vicuna-1.5) chat template that the official repo uses."""
    return (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's "
        "questions. USER: <image>\n" + question + " ASSISTANT:"
    )
