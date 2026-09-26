"""AMBER mass-tail diagnostic (rebuttal): SHAP-H distribution on the
generative and discriminative(existence) tasks of AMBER, for any backbone.

Mirrors scripts/sbc_h_distribution.py (same measurement path as the SBC
decoder: Mask2Former segments -> greedy lookahead -> LOO phi -> normalized
entropy H -> v3 route), but wrapper-agnostic (build_wrapper) and reading
AMBER queries instead of POPE.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/amber_h_distribution.py \
      --model llava-hf/llava-1.5-7b-hf --n 300 \
      --out results/amber_h_dist_llava7b.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.common import load_config, PROJECT_ROOT           # noqa: E402
from src.utils.segmentation import PanopticSegmenter             # noqa: E402
from src.decoding.ours_sbc import (_lookahead_with_logp,         # noqa: E402
                                   _blank_span_matches,
                                   _norm_entropy)
from src.decoding.ours_msb import _shap_phis_batched             # noqa: E402
from src.benchmarks.pope import POPE_QUESTION_SUFFIX             # noqa: E402

AMBER_DATA = PROJECT_ROOT / "data/AMBER/data"
AMBER_IMG = PROJECT_ROOT / "data/AMBER/image_extract/image"

# Same uniform resize as run_amber.py (see comment there): long side <= 640px.
MAX_SIDE = 640


def load_image(path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max(img.size) > MAX_SIDE:
        s = MAX_SIDE / max(img.size)
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)),
                         Image.LANCZOS)
    return img

TAU_MID = 0.5
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
BIN_NAMES = ["over-conc", "conc", "mixed", "spread", "over-spread"]


@torch.no_grad()
def measure(wrapper, segmenter, image, question):
    enc = wrapper.prepare_inputs(image, question)
    iid = enc["input_ids"]; pv = enc["pixel_values"]
    am = enc.get("attention_mask")
    segs = segmenter.segment(image, min_area_frac=0.01, max_segments=6)
    K = len(segs)
    if K < 2:
        return {"route": "fallback", "reason": "<2 seg", "K": K}
    span, *_ = _lookahead_with_logp(wrapper, iid, pv, am, True, 8,
                                    max_steps=32)
    if not span:
        return {"route": "fallback", "reason": "no span", "K": K}
    phis = _shap_phis_batched(wrapper, image, segs, iid, pv, am, span)
    H = _norm_entropy(phis)
    out = {"H": float(H), "K": K, "span_len": len(span),
           "phis": [float(x) for x in phis]}
    if H >= TAU_MID:
        match, *_ = _blank_span_matches(
            wrapper, iid, torch.zeros_like(pv), am, span, True, 8,
            max_steps=32)
        out["blank_match"] = bool(match)
        out["route"] = "pmi" if match else "msb"
    else:
        out["blank_match"] = None
        out["route"] = "msb"
    return out


def binned(hs):
    c = Counter()
    for h in hs:
        for (lo, hi), name in zip(BINS, BIN_NAMES):
            if lo <= h < hi:
                c[name] += 1
                break
    return {n: c.get(n, 0) for n in BIN_NAMES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tasks", nargs="+", default=["gen", "disc"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(None)
    if args.model is not None:
        cfg["model"]["name"] = args.model

    from src.models import build_wrapper
    print(f"loading model: {cfg['model']['name']} ...", flush=True)
    wrapper = build_wrapper(
        model_name=cfg["model"]["name"],
        dtype=getattr(torch, cfg["model"]["dtype"]),
        attn_implementation=cfg["model"]["attn_implementation"])
    seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                            dtype=getattr(torch, cfg["model"]["dtype"]))

    report = {"model": cfg["model"]["name"], "tau_mid": TAU_MID,
              "n_requested": args.n, "tasks": {}}
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for task in args.tasks:
        if task == "gen":
            qs = json.loads(
                (AMBER_DATA / "query/query_generative.json").read_text())
            suffix = ""
        else:
            qs = json.loads(
                (AMBER_DATA /
                 "query/query_discriminative-existence.json").read_text())
            suffix = POPE_QUESTION_SUFFIX
        idx = rng.choice(len(qs), size=min(args.n, len(qs)), replace=False)
        rows, routes = [], Counter()
        t0 = time.time()
        for k, i in enumerate(idx):
            q = qs[int(i)]
            img = load_image(AMBER_IMG / q["image"])
            r = measure(wrapper, seg, img, q["query"] + suffix)
            r.update(image=q["image"], query=q["query"], qid=q["id"])
            rows.append(r)
            routes[r["route"]] += 1
            if (k + 1) % 25 == 0:
                hs = [x["H"] for x in rows if "H" in x]
                el = time.time() - t0
                eta = el / (k + 1) * (len(idx) - k - 1)
                print(f"  [{task}] {k+1}/{len(idx)} routes={dict(routes)}"
                      f" H_mean={np.mean(hs):.3f}"
                      f" elapsed={el/60:.1f}m eta={eta/60:.1f}m", flush=True)
        jsonl = out_path.with_suffix("") .as_posix() + f"_{task}.jsonl"
        with open(jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        hs = [x["H"] for x in rows if "H" in x]
        b = binned(hs)
        tot = max(1, sum(b.values()))
        dom = max(b, key=b.get)
        report["tasks"][task] = {
            "n": len(rows), "n_with_H": len(hs),
            "routes": dict(routes),
            "H_mean": float(np.mean(hs)) if hs else None,
            "H_median": float(np.median(hs)) if hs else None,
            "bins": b,
            "dominant_tail": dom,
            "dominant_share": b[dom] / tot,
        }
        print(f"  -> {task}: n={len(rows)} bins={b} dom={dom}"
              f" ({b[dom]/tot:.1%})", flush=True)
        out_path.write_text(json.dumps(report, indent=2))

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
