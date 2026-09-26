"""Tiny smoke test for OPERA-Greedy: 5 CHAIR images + 5 POPE questions.

Verifies:
  - Module imports and dispatchers wire up correctly.
  - One full decode finishes (no shape / cache mismatch).
  - Output text is non-empty and grossly reasonable.

Usage:  CUDA_VISIBLE_DEVICES=1 python scripts/smoke_opera.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.llava_wrapper import LlavaWrapper            # noqa: E402
from src.utils.common import load_config                     # noqa: E402
from src.decoding.opera import opera_greedy_decode           # noqa: E402
from src.decoding.baseline import greedy_decode              # noqa: E402
from src.benchmarks.chair import CHAIR_PROMPT, sample_image_ids  # noqa: E402
from src.benchmarks.pope import POPE_QUESTION_SUFFIX         # noqa: E402


def main():
    cfg = load_config()
    w = LlavaWrapper(model_name=cfg["model"]["name"],
                     dtype=getattr(torch, cfg["model"]["dtype"]),
                     attn_implementation=cfg["model"]["attn_implementation"])

    op = cfg["opera"]
    img_dir = ROOT / cfg["benchmarks"]["chair"]["image_dir"]
    pope_qs = [json.loads(l) for l in open(
        ROOT / cfg["benchmarks"]["pope"]["data_dir"] / "coco_pope_random.json")][:5]

    # CHAIR — 2 imgs, max_new=64 (short to keep smoke quick)
    ids = sample_image_ids(img_dir, 2, cfg["benchmarks"]["chair"]["seed"])
    print("\n=== CHAIR smoke ===", flush=True)
    for cid, fn in ids:
        img = Image.open(img_dir / fn).convert("RGB")
        t0 = time.time()
        base = greedy_decode(w, img, CHAIR_PROMPT, max_new_tokens=64)
        t_base = time.time() - t0
        t0 = time.time()
        out = opera_greedy_decode(w, img, CHAIR_PROMPT, max_new_tokens=64,
                                  alpha=op["alpha"], sigma=op["sigma"],
                                  k_window=op["k_window"], ncan=op["ncan"])
        t_op = time.time() - t0
        print(f"  {fn}  greedy={t_base:.1f}s  opera={t_op:.1f}s  "
              f"(slowdown {t_op/max(t_base,1e-6):.1f}x)", flush=True)
        print(f"    BASE: {base[:140]!r}")
        print(f"    OPER: {out[:140]!r}")

    # POPE — 5 yes/no
    print("\n=== POPE smoke ===", flush=True)
    for q in pope_qs:
        img = Image.open(ROOT / cfg["benchmarks"]["pope"]["image_dir"] /
                         q["image"]).convert("RGB")
        full_q = q["text"] + POPE_QUESTION_SUFFIX
        t0 = time.time()
        base = greedy_decode(w, img, full_q, max_new_tokens=8)
        t_base = time.time() - t0
        t0 = time.time()
        out = opera_greedy_decode(w, img, full_q, max_new_tokens=8,
                                  alpha=op["alpha"], sigma=op["sigma"],
                                  k_window=op["k_window"], ncan=op["ncan"])
        t_op = time.time() - t0
        print(f"  q={q['text']!r}  gt={q.get('label')}  "
              f"greedy={t_base:.1f}s→{base!r}  opera={t_op:.1f}s→{out!r}",
              flush=True)


if __name__ == "__main__":
    main()
