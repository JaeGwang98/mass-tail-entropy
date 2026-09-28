"""Rebuttal: wall-clock decode overhead per method on CHAIR (long-form).

Complements results/diagnostics_latency.csv (POPE, 1-token answers) with
per-token cost on 512-token caption generation, where SBC's per-sentence
(not per-token) diagnostic amortizes.

For each method: decode N CHAIR images (COCO val2014, same sampler as the
benchmark) and report mean sec/image, sec/generated-token, tokens/image.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/measure_latency_chair.py \
      --n 20 --out results/diagnostics_latency_chair.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.common import load_config, set_seed, PROJECT_ROOT  # noqa: E402
from src.benchmarks.chair import (make_decoder, CHAIR_PROMPT,     # noqa: E402
                                  sample_image_ids)

SEG_METHODS = ("ours_sbc", "ours_sbc_v2", "ours_msb_sent", "ours_pmi_guard",
               "ours_lazy", "ours_lazy_attn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default=None)
    ap.add_argument("--methods", nargs="+",
                    default=["baseline", "vcd", "aif",
                             "opera", "ours_msb_sent", "ours_pmi",
                             "ours_sbc", "ours_lazy", "ours_lazy_attn"])
    ap.add_argument("--out", default="results/diagnostics_latency_chair.csv")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    cfg = load_config(None)
    if args.model is not None:
        cfg["model"]["name"] = args.model
    cfg["ours"]["msb_boost_factor"] = 1.8
    cfg["ours"]["image_margin_thresh"] = 0.5
    cfg["ours"]["sbc_tau_mid"] = 0.5

    from src.models import build_wrapper
    print(f"loading model: {cfg['model']['name']} ...", flush=True)
    wrapper = build_wrapper(
        model_name=cfg["model"]["name"],
        dtype=getattr(torch, cfg["model"]["dtype"]),
        attn_implementation=cfg["model"]["attn_implementation"])
    seg = None
    if any(m in SEG_METHODS for m in args.methods):
        from src.utils.segmentation import PanopticSegmenter
        seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                                dtype=getattr(torch, cfg["model"]["dtype"]))

    image_dir = PROJECT_ROOT / cfg["benchmarks"]["chair"]["image_dir"]
    ids = sample_image_ids(image_dir, args.n, args.seed)

    rows = []
    for m in args.methods:
        set_seed(args.seed)
        dec = make_decoder(m, wrapper, cfg, segmenter=seg)
        # warm-up on 1 image (exclude segmenter/model lazy init from timing)
        img0 = Image.open(image_dir / ids[0][1]).convert("RGB")
        _ = dec(img0, CHAIR_PROMPT)
        t0 = time.perf_counter()
        n_tok = 0
        for _, fname in ids:
            img = Image.open(image_dir / fname).convert("RGB")
            out = dec(img, CHAIR_PROMPT)
            if isinstance(out, tuple):
                out = out[0]
            n_tok += len(wrapper.tokenizer.encode(out,
                                                  add_special_tokens=False))
        sec = time.perf_counter() - t0
        rows.append((m, len(ids), sec / len(ids), sec / max(1, n_tok),
                     n_tok / len(ids)))
        print(f"{m:18s} N={len(ids)} sec/img={sec/len(ids):7.2f} "
              f"sec/tok={sec/max(1,n_tok):6.4f} tok/img={n_tok/len(ids):6.1f}",
              flush=True)
        torch.cuda.empty_cache()

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("method,n,sec_per_image,sec_per_token,tokens_per_image\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
