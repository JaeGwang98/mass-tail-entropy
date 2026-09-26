"""Verify that the PMI-prefill-reuse change in ours_sbc is *exactly*
result-preserving: run ours_sbc_decode with the reuse path active (deepcopied
prefill snapshots) and with it disabled (_kv_clone forced to None -> old
re-prefill path), and assert the decoded strings are byte-identical.  Also
times both to show the saving.

Usage:  CUDA_VISIBLE_DEVICES=0 python scripts/verify_sbc_prefill_reuse.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.llava_wrapper import LlavaWrapper              # noqa: E402
from src.utils.segmentation import PanopticSegmenter           # noqa: E402
from src.utils.common import load_config                       # noqa: E402
import src.decoding.ours_sbc as sbc                            # noqa: E402
from src.benchmarks.pope import POPE_QUESTION_SUFFIX          # noqa: E402
from src.benchmarks.chair import CHAIR_PROMPT, sample_image_ids  # noqa: E402


def _t():
    torch.cuda.synchronize(); return time.perf_counter()


@torch.no_grad()
def run_one(w, seg, img, q, mnt):
    return sbc.ours_sbc_decode(w, seg, img, q, max_new_tokens=mnt,
                               return_route=True)


@torch.no_grad()
def main():
    cfg = load_config()
    w = LlavaWrapper(model_name=cfg["model"]["name"],
                     dtype=getattr(torch, cfg["model"]["dtype"]),
                     attn_implementation=cfg["model"]["attn_implementation"])
    seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                            dtype=getattr(torch, cfg["model"]["dtype"]))
    pope_dir = ROOT / cfg["benchmarks"]["pope"]["image_dir"]
    chair_dir = ROOT / cfg["benchmarks"]["chair"]["image_dir"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pope", type=int, default=12)
    ap.add_argument("--n-chair", type=int, default=4)
    ap.add_argument("--shuffle-pope", action="store_true",
                    help="sample POPE questions at random (seed 42) instead of head")
    args = ap.parse_args()

    cases = []
    pope_qs = [json.loads(l) for l in open(
        ROOT / cfg["benchmarks"]["pope"]["data_dir"] / "coco_pope_random.json")]
    if args.shuffle_pope:
        random.Random(42).shuffle(pope_qs)
    for x in pope_qs[:args.n_pope]:
        cases.append(("POPE", pope_dir, x["image"], x["text"] + POPE_QUESTION_SUFFIX,
                      cfg["benchmarks"]["pope"]["max_new_tokens"]))
    if args.n_chair > 0:
        for cid, fn in sample_image_ids(chair_dir, args.n_chair,
                                        cfg["benchmarks"]["chair"]["seed"]):
            cases.append(("CHAIR", chair_dir, fn, CHAIR_PROMPT,
                          cfg["benchmarks"]["chair"]["max_new_tokens"]))

    _orig_clone = sbc._kv_clone
    n = mism = 0
    t_reuse = t_naive = 0.0
    routes = {}
    for tag, d, fn, q, mnt in cases:
        img = Image.open(d / fn).convert("RGB")

        sbc._kv_clone = _orig_clone                       # reuse path on
        t0 = _t(); out_r, route_r = run_one(w, seg, img, q, mnt); t_reuse += _t() - t0

        sbc._kv_clone = lambda _pkv: None                 # reuse path off (re-prefill)
        t0 = _t(); out_n, route_n = run_one(w, seg, img, q, mnt); t_naive += _t() - t0

        sbc._kv_clone = _orig_clone
        n += 1
        routes[route_r] = routes.get(route_r, 0) + 1
        ok = (out_r == out_n) and (route_r == route_n)
        if not ok:
            mism += 1
        flag = "" if ok else "  <<< MISMATCH"
        short = (out_r[:50] + "…") if len(out_r) > 50 else out_r
        print(f"[{tag}] {fn:40s} route={route_r:9s} | {short!r}{flag}")
        if not ok:
            print(f"    reuse : {out_r!r}")
            print(f"    naive : {out_n!r}")

    print(f"\n==> {n} cases | routes={routes} | mismatches: {mism}/{n}")
    print(f"    wall time   reuse path: {t_reuse/n*1000:8.1f} ms/case   "
          f"re-prefill path: {t_naive/n*1000:8.1f} ms/case   "
          f"saved {(1 - t_reuse/max(t_naive,1e-9))*100:.1f}%")
    print("PASS — prefill reuse is byte-identical to re-prefilling"
          if mism == 0 else "FAIL — outputs differ; inspect above")


if __name__ == "__main__":
    main()
