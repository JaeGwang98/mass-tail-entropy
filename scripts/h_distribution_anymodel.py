"""Rebuttal (R1/R2 breadth): mass-tail H diagnostic on POPE + CHAIR for any
backbone (third-backbone check: Qwen2-VL-7B = architecture axis, LLaVA-13B
8-bit = scale axis; the diagnostic uses forward passes only, so quantization
caveats about attention-magnitude actuators do not apply).

Same measurement path as the SBC decoder (Mask2Former segments -> greedy
lookahead -> LOO phi -> normalized entropy H -> v3 route), mirroring
scripts/amber_h_distribution.py.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/h_distribution_anymodel.py \
      --model Qwen/Qwen2-VL-7B-Instruct \
      --out results/h_dist_qwen2vl.json
  CUDA_VISIBLE_DEVICES=1 python scripts/h_distribution_anymodel.py \
      --model llava-hf/llava-1.5-13b-hf --load-8bit \
      --out results/h_dist_llava13b_8bit.json
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
from src.benchmarks.chair import CHAIR_PROMPT, sample_image_ids  # noqa: E402

TAU_MID = 0.5
BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
BIN_NAMES = ["over-conc", "conc", "mixed", "spread", "over-spread"]


@torch.no_grad()
def measure(wrapper, segmenter, image, question, max_segments=6):
    enc = wrapper.prepare_inputs(image, question)
    iid = enc["input_ids"]; pv = enc["pixel_values"]
    am = enc.get("attention_mask")
    segs = segmenter.segment(image, min_area_frac=0.01,
                             max_segments=max_segments)
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


def iter_pope(cfg, n_per_split, rng):
    data_dir = PROJECT_ROOT / cfg["benchmarks"]["pope"]["data_dir"]
    img_dir = PROJECT_ROOT / cfg["benchmarks"]["pope"]["image_dir"]
    for split in ("random", "popular", "adversarial"):
        qs = [json.loads(l)
              for l in open(data_dir / f"coco_pope_{split}.json") if l.strip()]
        idx = rng.choice(len(qs), size=min(n_per_split, len(qs)),
                         replace=False)
        for i in idx:
            q = qs[int(i)]
            p = img_dir / q["image"]
            if p.exists():
                yield (f"pope-{split}", q["image"],
                       q["text"] + POPE_QUESTION_SUFFIX, p)


def iter_chair(cfg, n, seed):
    img_dir = PROJECT_ROOT / cfg["benchmarks"]["chair"]["image_dir"]
    for _, fname in sample_image_ids(img_dir, n, seed):
        yield ("chair", fname, CHAIR_PROMPT, img_dir / fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--n-pope", type=int, default=200,
                    help="questions per POPE split (3 splits)")
    ap.add_argument("--n-chair", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--segmenter", default="m2f", choices=["m2f", "sam"])
    ap.add_argument("--max-segments", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config(None)
    cfg["model"]["name"] = args.model

    from src.models import build_wrapper
    print(f"loading model: {args.model} (8bit={args.load_8bit}) ...",
          flush=True)
    wrapper = build_wrapper(
        model_name=args.model,
        dtype=getattr(torch, cfg["model"]["dtype"]),
        attn_implementation=cfg["model"]["attn_implementation"],
        load_in_8bit=args.load_8bit)
    if args.segmenter == "sam":
        from src.utils.segmentation import SAMSegmenter
        seg = SAMSegmenter(dtype=getattr(torch, cfg["model"]["dtype"]))
    else:
        seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                                dtype=getattr(torch, cfg["model"]["dtype"]))

    rng = np.random.default_rng(args.seed)
    sources = list(iter_chair(cfg, args.n_chair,
                              cfg["benchmarks"]["chair"]["seed"]))
    sources += list(iter_pope(cfg, args.n_pope, rng))

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_task = {}
    t0 = time.time()
    jsonl_path = out_path.with_suffix("").as_posix() + "_rows.jsonl"
    with open(jsonl_path, "w") as fj:
        for k, (task, img_name, question, path) in enumerate(sources):
            img = Image.open(path).convert("RGB")
            r = measure(wrapper, seg, img, question,
                        max_segments=args.max_segments)
            r.update(task=task, image=img_name)
            fj.write(json.dumps(r) + "\n")
            per_task.setdefault(task, []).append(r)
            if (k + 1) % 25 == 0:
                fj.flush()
                el = time.time() - t0
                eta = el / (k + 1) * (len(sources) - k - 1)
                print(f"  {k+1}/{len(sources)} ({task})"
                      f" elapsed={el/60:.1f}m eta={eta/60:.1f}m", flush=True)

    report = {"model": args.model, "load_8bit": args.load_8bit,
              "segmenter": args.segmenter, "max_segments": args.max_segments,
              "tau_mid": TAU_MID, "tasks": {}}
    # pope-* splits pooled as well, matching the paper's POPE cell
    pooled = [r for t, rows in per_task.items() if t.startswith("pope")
              for r in rows]
    for task, rows in list(per_task.items()) + [("pope-all", pooled)]:
        hs = [x["H"] for x in rows if "H" in x]
        b = binned(hs)
        tot = max(1, sum(b.values()))
        dom = max(b, key=b.get)
        report["tasks"][task] = {
            "n": len(rows), "n_with_H": len(hs), "bins": b,
            "H_mean": float(np.mean(hs)) if hs else None,
            "dominant_tail": dom, "dominant_share": b[dom] / tot,
            "routes": dict(Counter(x["route"] for x in rows)),
        }
        print(f"  -> {task}: n={len(rows)} dom={dom} ({b[dom]/tot:.1%})"
              f" bins={b}", flush=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
