"""AMBER benchmark runner (rebuttal: mixed-format generalization).

Generative task  : 1004 images, prompt from query_generative.json
                   ("Describe this image."), chair-style decoding
                   (max_new_tokens from chair config).
Discriminative   : yes/no questions (default: existence subset, 4924q,
                   the hallucination-targeted split), pope-style decoding
                   (POPE_QUESTION_SUFFIX + parse_yes_no).

Ground truth: data/AMBER/data/annotations.json ("truth"/"hallu" for
generative ids, "truth": yes/no for discriminative ids).

Outputs (per run, under --out-dir):
  raw_<method>.jsonl        one record per query
  summary_<method>.json     metrics + timing
  amber_resp_<method>.json  official AMBER format [{"id", "response"}]
                            for data/AMBER/inference.py

Usage examples:
  python scripts/run_amber.py --task disc --method ours_sbc \
      --out-dir results/amber_disc_sbc_llava7b
  python scripts/run_amber.py --task gen --method baseline_greedy \
      --model Qwen/Qwen2.5-VL-7B-Instruct \
      --out-dir results/amber_gen_baseline_qwen2_5vl
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

from src.utils.common import load_config, set_seed, PROJECT_ROOT   # noqa: E402
from src.benchmarks.pope import (make_decoder as pope_make_decoder,  # noqa: E402
                                 parse_yes_no, compute_metrics,
                                 POPE_QUESTION_SUFFIX)
from src.benchmarks.chair import make_decoder as chair_make_decoder  # noqa: E402
from src.benchmarks.amber import AMBERGen                           # noqa: E402

AMBER_DATA = PROJECT_ROOT / "data/AMBER/data"
AMBER_IMG = PROJECT_ROOT / "data/AMBER/image_extract/image"

# 160/1004 AMBER images exceed 1MP (max 54MP). Resize long side to 640px
# (COCO val2014 scale) uniformly for ALL backbones: Qwen2.5-VL's dynamic
# resolution would otherwise produce ~30k visual tokens (OOM under eager
# attention); LLaVA resizes to 336px internally either way.
MAX_SIDE = 640


def load_image(path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max(img.size) > MAX_SIDE:
        s = MAX_SIDE / max(img.size)
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)),
                         Image.LANCZOS)
    return img

SEG_METHODS = ("ours_sbc", "ours_sbc_v2", "ours_msb_sent", "ours_pmi_guard",
               "ours_no_h", "ours_logit_h", "ours_attn_h",
               "ours_task_pmi", "ours_task_msb", "ours_lazy", "ours_lazy_attn")


def load_queries(task: str, disc_file: str):
    if task == "gen":
        return json.loads((AMBER_DATA / "query/query_generative.json").read_text())
    name = {"existence": "query_discriminative-existence.json",
            "attribute": "query_discriminative-attribute.json",
            "relation": "query_discriminative-relation.json",
            "all": "query_discriminative.json"}[disc_file]
    return json.loads((AMBER_DATA / "query" / name).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gen", "disc"])
    ap.add_argument("--method", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--disc-file", default="existence",
                    choices=["existence", "attribute", "relation", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--boost-factor", type=float, default=None)
    ap.add_argument("--image-margin-thresh", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model is not None:
        cfg["model"]["name"] = args.model
        print(f"[override] model = {args.model}")
    if args.boost_factor is not None:
        cfg.setdefault("ours", {})["msb_boost_factor"] = args.boost_factor
    if args.image_margin_thresh is not None:
        cfg.setdefault("ours", {})["image_margin_thresh"] = args.image_margin_thresh

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.models import build_wrapper
    print(f"loading model: {cfg['model']['name']} ...", flush=True)
    wrapper = build_wrapper(
        model_name=cfg["model"]["name"],
        dtype=getattr(torch, cfg["model"]["dtype"]),
        attn_implementation=cfg["model"]["attn_implementation"])
    segmenter = None
    if args.method in SEG_METHODS:
        from src.utils.segmentation import PanopticSegmenter
        print("loading Mask2Former...", flush=True)
        segmenter = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                                      dtype=getattr(torch, cfg["model"]["dtype"]))

    queries = load_queries(args.task, args.disc_file)
    if args.limit:
        queries = queries[: args.limit]

    ann = json.loads((AMBER_DATA / "annotations.json").read_text())
    truth_by_id = {a["id"]: a for a in ann}

    if args.task == "disc":
        decoder = pope_make_decoder(args.method, wrapper, cfg,
                                    segmenter=segmenter)
    else:
        # chair.make_decoder uses "baseline"/"vcd" (greedy is the chair default)
        gen_alias = {"baseline_greedy": "baseline", "vcd_greedy": "vcd"}
        args.method = gen_alias.get(args.method, args.method)
        decoder = chair_make_decoder(args.method, wrapper, cfg,
                                     segmenter=segmenter)

    set_seed(args.seed)
    raw_path = out_dir / f"raw_{args.method}.jsonl"
    rows = []
    t0 = time.time()
    with open(raw_path, "w") as fraw:
        for i, q in enumerate(queries):
            img = load_image(AMBER_IMG / q["image"])
            question = q["query"] + (POPE_QUESTION_SUFFIX
                                     if args.task == "disc" else "")
            result = decoder(img, question)
            route = None
            if isinstance(result, tuple):
                result, route = result
            rec = {"id": q["id"], "image": q["image"], "query": q["query"],
                   "response": result}
            if route is not None:
                rec["route"] = route
            if args.task == "disc":
                rec["truth"] = truth_by_id[q["id"]]["truth"]
                rec["pred"] = parse_yes_no(result)
            fraw.write(json.dumps(rec) + "\n")
            if (i + 1) % 50 == 0:
                fraw.flush()
                el = time.time() - t0
                eta = el / (i + 1) * (len(queries) - i - 1)
                print(f"  [{args.task}/{args.method}] {i+1}/{len(queries)}"
                      f"  elapsed={el/60:.1f}m  eta={eta/60:.1f}m", flush=True)
            rows.append(rec)
    seconds = time.time() - t0

    # official AMBER format for data/AMBER/inference.py
    resp_path = out_dir / f"amber_resp_{args.method}.json"
    resp_path.write_text(json.dumps(
        [{"id": r["id"], "response": r["response"]} for r in rows], indent=1))

    summary = {"task": args.task, "method": args.method,
               "model": cfg["model"]["name"], "n": len(rows),
               "seconds": seconds,
               "disc_file": args.disc_file if args.task == "disc" else None}

    if args.task == "disc":
        m = compute_metrics([r["pred"] for r in rows],
                            [r["truth"] for r in rows])
        summary["metrics"] = m
        print(json.dumps(m, indent=2))
    else:
        matcher = AMBERGen(AMBER_DATA)
        n_sent_hallu = 0
        n_obj = n_hallu_obj = 0
        cover_sum = 0.0
        len_sum = 0
        for r in rows:
            objs = matcher.objects_in_caption(r["response"], r["id"])
            hallu = [o for o in objs if o[2]]
            n_obj += len(objs)
            n_hallu_obj += len(hallu)
            n_sent_hallu += 1 if hallu else 0
            tset = matcher._truth_comp.get(r["id"], set())
            mentioned = {matcher._find(o[0]) for o in objs if not o[2]}
            cover_sum += len(mentioned & tset) / max(1, len(tset))
            len_sum += len(r["response"].split())
        n = max(1, len(rows))
        summary["metrics"] = {
            "chair_s_like": n_sent_hallu / n,
            "chair_i_like": n_hallu_obj / max(1, n_obj),
            "coverage": cover_sum / n,
            "avg_len_words": len_sum / n,
            "n_obj_mentions": n_obj,
        }
        print(json.dumps(summary["metrics"], indent=2))

    (out_dir / f"summary_{args.method}.json").write_text(
        json.dumps(summary, indent=2))
    print(f"wrote {out_dir}  ({seconds/60:.1f} min)")


if __name__ == "__main__":
    main()
