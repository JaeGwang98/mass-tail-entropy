"""MME evaluator -- hallucination subset (existence / count / position / color).

Loads the ``lmms-lab/MME`` parquet shards from the local HuggingFace cache,
runs the chosen decoding method on the four object/attribute hallucination
subtasks, and reports the standard MME score (accuracy + accuracy+) per
subtask plus an overall accuracy/F1.

The MME questions already end with "Please answer yes or no", so no extra
prompt suffix is appended (unlike POPE).

Usage:
    python -m src.benchmarks.mme --method baseline_greedy \\
        --model llava-hf/llava-1.5-7b-hf --out-dir results/mme_baseline_llava7b
"""
from __future__ import annotations

import argparse
import gc
import glob
import io
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from ..utils.common import PROJECT_ROOT, load_config
from .pope import make_decoder, parse_yes_no

# MME hallucination subset -- the four object/attribute subtasks, following
# the VCD convention. The cognition subtasks (code/math/translation) are
# perception-irrelevant and excluded.
HALLUCINATION_SUBTASKS = ["existence", "count", "position", "color"]

MME_PARQUET_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--lmms-lab--MME/snapshots/*/"
    "data/*.parquet")


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_mme_subset() -> pd.DataFrame:
    files = sorted(glob.glob(MME_PARQUET_GLOB))
    if not files:
        raise FileNotFoundError(
            f"MME parquet shards not found at {MME_PARQUET_GLOB}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df["category"].isin(HALLUCINATION_SUBTASKS)].reset_index(drop=True)
    return df


def _to_pil(field) -> Image.Image:
    """The parquet 'image' column is an HF image struct {bytes, path}."""
    if isinstance(field, dict) and field.get("bytes") is not None:
        return Image.open(io.BytesIO(field["bytes"])).convert("RGB")
    if isinstance(field, (bytes, bytearray)):
        return Image.open(io.BytesIO(field)).convert("RGB")
    return field.convert("RGB")          # already a PIL image


# ---------------------------------------------------------------------------
# scoring -- standard MME metric
# ---------------------------------------------------------------------------
def score_mme(raw: list) -> dict:
    """raw: list of {category, qid, gt, pred, correct}.

    MME score per subtask = (accuracy + accuracy_plus) x 100, where
    accuracy_plus is the fraction of images for which BOTH questions are
    correct. Subtask scores sum to the hallucination-subset total."""
    per_cat = {}
    for cat in HALLUCINATION_SUBTASKS:
        rows = [r for r in raw if r["category"] == cat]
        if not rows:
            continue
        acc = sum(r["correct"] for r in rows) / len(rows)
        by_img = defaultdict(list)
        for r in rows:
            by_img[r["qid"]].append(r["correct"])
        acc_plus = sum(all(v) for v in by_img.values()) / max(len(by_img), 1)
        per_cat[cat] = {"n": len(rows), "n_images": len(by_img),
                        "acc": acc, "acc_plus": acc_plus,
                        "score": (acc + acc_plus) * 100.0}
    subset_score = sum(c["score"] for c in per_cat.values())
    n = len(raw)
    tp = sum(1 for r in raw if r["pred"] == "yes" and r["gt"] == "yes")
    tn = sum(1 for r in raw if r["pred"] == "no" and r["gt"] == "no")
    fp = sum(1 for r in raw if r["pred"] == "yes" and r["gt"] == "no")
    fn = sum(1 for r in raw if r["pred"] == "no" and r["gt"] == "yes")
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {"subset_score": subset_score,
            "overall_acc": (tp + tn) / max(1, n),
            "overall_f1": f1, "n": n, "per_category": per_cat}


# ---------------------------------------------------------------------------
# evaluation loop
# ---------------------------------------------------------------------------
def evaluate(method: str, wrapper, cfg: dict, segmenter=None,
             limit: int | None = None, out_dir: Path | None = None) -> dict:
    df = load_mme_subset()
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)
    decoder = make_decoder(method, wrapper, cfg, segmenter=segmenter)

    raw, t0 = [], time.time()
    for i, row in df.iterrows():
        img = _to_pil(row["image"])
        result = decoder(img, str(row["question"]))
        if isinstance(result, tuple):
            text, route = result
        else:
            text, route = result, None
        pred = parse_yes_no(text)
        gt = str(row["answer"]).strip().lower()
        raw.append({"qid": str(row["question_id"]),
                    "category": str(row["category"]),
                    "question": str(row["question"]), "gt": gt,
                    "pred": pred, "correct": pred == gt, "raw_text": text,
                    "route": route})
        if (i + 1) % 50 == 0:
            print(f"    [{method}] {i + 1}/{len(df)}", flush=True)
        if i % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    summary = score_mme(raw)
    summary["method"] = method
    summary["seconds"] = time.time() - t0
    if out_dir is not None:
        (out_dir / f"raw_{method}.jsonl").write_text(
            "\n".join(json.dumps(x) for x in raw))
        (out_dir / f"summary_{method}.json").write_text(
            json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="results/mme")
    ap.add_argument("--boost-factor", type=float, default=None)
    ap.add_argument("--image-margin-thresh", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of questions (debug).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.boost_factor is not None:
        cfg.setdefault("ours", {})["msb_boost_factor"] = args.boost_factor
        print(f"[override] msb_boost_factor = {args.boost_factor}")
    if args.image_margin_thresh is not None:
        cfg.setdefault("ours", {})["image_margin_thresh"] = args.image_margin_thresh
        print(f"[override] image_margin_thresh = {args.image_margin_thresh}")
    if args.model is not None:
        cfg["model"]["name"] = args.model
        print(f"[override] model = {args.model}")
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from ..models import build_wrapper
    print(f"loading model: {cfg['model']['name']} ...")
    wrapper = build_wrapper(
        model_name=cfg["model"]["name"],
        dtype=getattr(torch, cfg["model"]["dtype"]),
        attn_implementation=cfg["model"]["attn_implementation"])
    segmenter = None
    if args.method in ("ours_sbc", "ours_sbc_v2", "ours_msb_sent", "ours_pmi_guard",
                       "ours_no_h", "ours_lazy", "ours_lazy_attn"):
        from ..utils.segmentation import PanopticSegmenter
        print("loading Mask2Former...")
        segmenter = PanopticSegmenter(
            model_name=cfg["mask2former"]["name"],
            dtype=getattr(torch, cfg["model"]["dtype"]))

    print(f"\n=== {args.method} on MME (hallucination subset) ===")
    s = evaluate(args.method, wrapper, cfg, segmenter=segmenter,
                 limit=args.limit, out_dir=out_dir)
    print(f"  subset MME score = {s['subset_score']:.1f}   "
          f"overall acc = {s['overall_acc'] * 100:.2f}   "
          f"F1 = {s['overall_f1'] * 100:.2f}")
    for cat, c in s["per_category"].items():
        print(f"   {cat:10s} acc {c['acc'] * 100:5.1f} / "
              f"acc+ {c['acc_plus'] * 100:5.1f} / score {c['score']:6.1f}  "
              f"(n={c['n']})")


if __name__ == "__main__":
    main()
