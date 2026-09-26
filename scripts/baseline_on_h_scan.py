"""Companion to sbc_h_distribution.py — runs baseline greedy on the EXACT
same POPE qids that were sampled by the H scan, then emits a per-row JSONL
with {qid, H, route, baseline_pred, gt, baseline_correct}.

Reads:  results/sbc_h_dist_pope_{split}.jsonl       (H rows from the scan)
Writes: results/sbc_h_baseline_{split}.jsonl
        results/sbc_h_baseline_summary.json

Usage:  CUDA_VISIBLE_DEVICES=1 python scripts/baseline_on_h_scan.py
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
from src.models.llava_wrapper import LlavaWrapper                # noqa: E402
from src.utils.common import load_config                         # noqa: E402
from src.benchmarks.pope import POPE_QUESTION_SUFFIX, parse_yes_no  # noqa: E402


@torch.no_grad()
def baseline_pred(w, image, question, max_new_tokens=8):
    """One greedy decode capped at 8 tokens (enough for 'Yes' / 'No.')."""
    enc = w.prepare_inputs(image, question)
    out = w.prefill(enc["input_ids"], enc["pixel_values"],
                    enc.get("attention_mask"))
    logits = out.logits[:, -1, :]; pkv = out.past_key_values
    eos = int(w.tokenizer.eos_token_id)
    toks = []
    for _ in range(max_new_tokens):
        nxt = logits.argmax(-1, keepdim=True); t = int(nxt.item())
        if t == eos:
            break
        toks.append(t)
        logits, pkv = w.decode_step(nxt, pkv)
        logits = logits[:, -1, :]
    return w.tokenizer.decode(toks).strip()


def main():
    cfg = load_config()
    img_dir = ROOT / cfg["benchmarks"]["pope"]["image_dir"]
    print("loading LLaVA...", flush=True)
    w = LlavaWrapper(model_name=cfg["model"]["name"],
                     dtype=getattr(torch, cfg["model"]["dtype"]),
                     attn_implementation=cfg["model"]["attn_implementation"])

    summary = {}
    for split in ("random", "popular", "adversarial"):
        in_path = ROOT / "results" / f"sbc_h_dist_pope_{split}.jsonl"
        if not in_path.exists():
            print(f"  ! missing {in_path}, skipping", flush=True); continue
        rows = [json.loads(l) for l in open(in_path)]
        out_rows = []
        t0 = time.time()
        for k, r in enumerate(rows):
            img = Image.open(img_dir / r["image"]).convert("RGB")
            ans = baseline_pred(w, img, r["question"] + POPE_QUESTION_SUFFIX)
            pred = parse_yes_no(ans)
            gt = str(r.get("gt", "")).strip().lower()
            out_rows.append({"qid": r["qid"], "image": r["image"],
                             "question": r["question"], "gt": gt,
                             "H": r.get("H"), "route": r.get("route"),
                             "K": r.get("K"), "span_len": r.get("span_len"),
                             "baseline_raw": ans, "baseline_pred": pred,
                             "baseline_correct": pred == gt})
            if (k + 1) % 50 == 0:
                acc = sum(x["baseline_correct"] for x in out_rows) / len(out_rows)
                print(f"  [{split}] {k+1}/{len(rows)}  acc_so_far={acc:.3f}  "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        out = ROOT / "results" / f"sbc_h_baseline_{split}.jsonl"
        with open(out, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        acc = sum(x["baseline_correct"] for x in out_rows) / max(1, len(out_rows))
        summary[split] = {"n": len(out_rows), "baseline_accuracy": acc}
        print(f"  → {split}: n={len(out_rows)}  baseline_acc={acc:.3f}",
              flush=True)
    (ROOT / "results" / "sbc_h_baseline_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\n✓ done. summary: {summary}")


if __name__ == "__main__":
    main()
