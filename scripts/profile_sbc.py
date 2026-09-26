"""Profile where ours_sbc_decode spends its time, per phase, for POPE and CHAIR.

Usage:  CUDA_VISIBLE_DEVICES=0 python scripts/profile_sbc.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.llava_wrapper import LlavaWrapper                # noqa: E402
from src.utils.segmentation import PanopticSegmenter             # noqa: E402
from src.utils.common import load_config                         # noqa: E402
from src.decoding.ours_v3 import _greedy_lookahead              # noqa: E402
from src.decoding.ours_msb_rolling import _greedy_lookahead_until_period  # noqa: E402
from src.decoding.ours_msb import _shap_phis_batched, _topk_diverse  # noqa: E402
from src.decoding.ours_v4 import (_segment_to_visual_token_indices,  # noqa: E402
                                  _build_boost_mask, _build_step_boost_mask)
from src.decoding.vcd import _apc_filter                        # noqa: E402
from src.decoding.ours_sbc import _norm_entropy                 # noqa: E402
from src.benchmarks.pope import POPE_QUESTION_SUFFIX            # noqa: E402
from src.benchmarks.chair import CHAIR_PROMPT, sample_image_ids  # noqa: E402


def _t():
    torch.cuda.synchronize(); return time.perf_counter()


@torch.no_grad()
def profile_one(w, seg, img, question, sentence, max_new):
    times = {}
    t0 = _t()
    enc = w.prepare_inputs(img, question)
    input_ids, pixel_v, attn = enc["input_ids"], enc["pixel_values"], enc.get("attention_mask")
    times["prepare_inputs"] = _t() - t0
    t0 = _t(); segments = seg.segment(img, min_area_frac=0.01, max_segments=6); times["mask2former_segment"] = _t() - t0
    if len(segments) < 2:
        return None
    t0 = _t()
    span = (_greedy_lookahead_until_period(w, input_ids, pixel_v, attn, max_steps=32)
            if sentence else _greedy_lookahead(w, input_ids, pixel_v, attn, 1))
    times["lookahead_v"] = _t() - t0
    if not span:
        return None
    t0 = _t(); phis = _shap_phis_batched(w, img, segments, input_ids, pixel_v, attn, span); times["shap_phis"] = _t() - t0
    H = _norm_entropy(phis)
    t0 = _t()
    if H >= 0.5:
        span_b = (_greedy_lookahead_until_period(w, input_ids, torch.zeros_like(pixel_v), attn, max_steps=32)
                  if sentence else _greedy_lookahead(w, input_ids, torch.zeros_like(pixel_v), attn, 1))
        agree = (span_b == span)
    else:
        agree = False
    times["blank_lookahead"] = _t() - t0
    t0 = _t()
    if H >= 0.5 and agree:
        # PMI generation
        eos = int(w.tokenizer.eos_token_id)
        pb = torch.zeros_like(pixel_v)
        ov = w.prefill(input_ids, pixel_v, attn); ob = w.prefill(input_ids, pb, attn)
        lv = ov.logits[:, -1, :]; lb = ob.logits[:, -1, :]; pkv_v, pkv_b = ov.past_key_values, ob.past_key_values
        gen = []
        for _ in range(max_new):
            blend = 2.0 * lv - 1.0 * lb
            blend = blend.masked_fill(~_apc_filter(lv, 0.1), float("-inf"))
            nxt = blend.argmax(-1, keepdim=True)
            if int(nxt.item()) == eos: break
            gen.append(int(nxt.item()))
            lv, pkv_v = w.decode_step(nxt, pkv_v); lb, pkv_b = w.decode_step(nxt, pkv_b)
            lv = lv[:, -1, :]; lb = lb[:, -1, :]
        route = "pmi"
    else:
        # MSB generation
        eos = int(w.tokenizer.eos_token_id)
        vpos = w.visual_token_positions(input_ids); text_start = vpos[-1] + 1
        chosen = _topk_diverse(phis, segments, 2, 0.5)
        bpos = sorted(set(p for ci in chosen for p in _segment_to_visual_token_indices(segments[ci].mask, vpos)))
        if bpos:
            bm = _build_boost_mask(input_ids.shape[1], bpos, text_start, 1.5, w.device, w.dtype)
            out = w.prefill(input_ids, pixel_v, bm); logits = out.logits[:, -1, :]; pkv = out.past_key_values
            cur = input_ids.shape[1]; gen = []
            for _ in range(max_new):
                nxt = logits.argmax(-1, keepdim=True)
                if int(nxt.item()) == eos: break
                gen.append(int(nxt.item()))
                sm = _build_step_boost_mask(cur, bpos, 1.5, w.device, w.dtype)
                logits, pkv = w.decode_step(nxt, pkv, attention_mask=sm); logits = logits[:, -1, :]; cur += 1
        route = "msb"
    times["generation_" + route] = _t() - t0
    times["__route"] = route
    times["__K"] = len(segments)
    times["__span_len"] = len(span)
    return times


def main():
    cfg = load_config()
    w = LlavaWrapper(model_name=cfg["model"]["name"], dtype=getattr(torch, cfg["model"]["dtype"]),
                     attn_implementation=cfg["model"]["attn_implementation"])
    seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"], dtype=getattr(torch, cfg["model"]["dtype"]))
    pope_dir = ROOT / cfg["benchmarks"]["pope"]["image_dir"]
    chair_dir = ROOT / cfg["benchmarks"]["chair"]["image_dir"]

    for tag, items in [
        ("POPE", [("POPE", pope_dir, q["image"], q["text"] + POPE_QUESTION_SUFFIX, False)
                  for q in [json.loads(l) for l in open(ROOT / cfg["benchmarks"]["pope"]["data_dir"] / "coco_pope_random.json")][:30]]),
        ("CHAIR", [("CHAIR", chair_dir, fn, CHAIR_PROMPT, True)
                   for cid, fn in sample_image_ids(chair_dir, 8, cfg["benchmarks"]["chair"]["seed"])]),
    ]:
        mn = cfg["benchmarks"][tag.lower()]["max_new_tokens"]
        agg = defaultdict(float); cnt = 0; routes = defaultdict(int)
        for _, d, fn, q, sent in items:
            r = profile_one(w, seg, Image.open(d / fn).convert("RGB"), q, sent, mn)
            if r is None: continue
            cnt += 1; routes[r["__route"]] += 1
            for k, v in r.items():
                if not k.startswith("__"): agg[k] += v
        if cnt == 0:
            print(f"=== {tag}: no usable items ==="); continue
        total = sum(agg.values())
        print(f"\n=== {tag}  ({cnt} items, routes={dict(routes)}, max_new={mn}) — per-item averages ===")
        for k, v in sorted(agg.items(), key=lambda x: -x[1]):
            print(f"  {k:24s}: {v/cnt*1000:8.1f} ms/item   ({100*v/total:5.1f}%)")
        print(f"  {'TOTAL':24s}: {total/cnt*1000:8.1f} ms/item")


if __name__ == "__main__":
    main()
