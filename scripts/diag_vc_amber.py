r"""Object-level targeted v_c diagnostic on AMBER generative task (LLaVA).

Same decisive test as scripts/diag_vc_chair.py but on AMBER (richer: object +
attribute + relation hallucination, LLM-free matcher).  Tests:
  "A hallucinated object has little visual support (low v_c)?"
at the object level, with the first-mention refinement (context-repetition
removed).

Usage:
  python -m scripts.diag_vc_amber --limit 500 --cap-tokens 64
  python -m scripts.diag_vc_amber --limit 250 --offset 250   # GPU shard 2
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.utils.common import PROJECT_ROOT, load_config, set_seed
from src.utils.segmentation import PanopticSegmenter, mask_image_with_segments
from src.benchmarks.amber import AMBERGen
from src.decoding import exact_shap as ES
from scripts.diag_vc_chair import greedy_caption, token_char_offsets

AMBER_PROMPT = "Describe this image."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--cap-tokens", type=int, default=64)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--model", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--amber-dir", default="data/AMBER")
    ap.add_argument("--out-dir", default="results/exactshap")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model: cfg["model"]["name"] = args.model
    from src.models import build_wrapper
    print(f"loading {cfg['model']['name']} ...", flush=True)
    wrapper = build_wrapper(model_name=cfg["model"]["name"],
                            dtype=getattr(torch, cfg["model"]["dtype"]),
                            attn_implementation=cfg["model"]["attn_implementation"])
    print("loading Mask2Former ...", flush=True)
    seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                            dtype=getattr(torch, cfg["model"]["dtype"]))
    m2f = cfg["mask2former"]

    amber_dir = PROJECT_ROOT / args.amber_dir
    matcher = AMBERGen(amber_dir / "data")
    img_dir = amber_dir / "image_extract" / "image"
    ids = list(range(1, 1005))[args.offset:args.offset + args.limit]
    set_seed(args.seed)

    rows = []; t0 = time.time(); n_skip = 0
    for i, iid in enumerate(ids):
        try:
            img = Image.open(img_dir / f"AMBER_{iid}.jpg").convert("RGB")
            enc = wrapper.prepare_inputs(img, AMBER_PROMPT)
            cap_ids = greedy_caption(wrapper, enc, args.cap_tokens)
            if not cap_ids:
                n_skip += 1; continue
            offs, text = token_char_offsets(wrapper.tokenizer, cap_ids)
            objs = matcher.objects_in_caption(text, iid)
            if not objs:
                continue
            segs = seg.segment(img, min_area_frac=m2f["min_area_frac"],
                               max_segments=m2f["max_segments"])
            K = len(segs)
            if K < 2:
                n_skip += 1; continue
            subsets = ES.all_subsets(K)
            imgs = [mask_image_with_segments(img,
                        [segs[k].mask for k in range(K) if k not in S])
                    for S in subsets]
            mat = ES.coalition_pertoken_logp(wrapper, imgs, enc["input_ids"],
                                             cap_ids, max_batch=args.max_batch)
            for (canon, (cs, ce), hall) in objs:
                pos = [j for j, (a, b) in enumerate(offs) if a < ce and b > cs]
                if not pos:
                    continue
                vals = mat[:, pos].sum(axis=1)
                phi, v_bg, v_N, M = ES.exact_shap_from_coalition_values(
                    vals, subsets, K)
                phi_loo = ES.loo_from_coalition_values(vals, subsets, K)
                rows.append({
                    "iid": iid, "obj": canon, "hallucinated": bool(hall),
                    "n_tok": len(pos), "K": K,
                    "M_c": float(M), "v_bg_c": float(v_bg), "v_N_c": float(v_N),
                    "maxphi_c": float(phi.max()), "minphi_c": float(phi.min()),
                    "M_c_loo": float(phi_loo.sum()),
                    "maxphi_c_loo": float(phi_loo.max()),
                    "caption": text,
                })
        except Exception as e:
            n_skip += 1
            print(f"  [skip iid={iid}] {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            nh = sum(r["hallucinated"] for r in rows)
            print(f"  {i+1}/{len(ids)} objs={len(rows)} hallObj={nh} "
                  f"skip={n_skip} ({(time.time()-t0)/(i+1):.1f}s/img)", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    tag = cfg["model"]["name"].split("/")[-1]
    shard = "" if args.offset == 0 else f"_off{args.offset}"
    out_dir = (PROJECT_ROOT / args.out_dir /
               f"{tag}_amber_vc_greedy_seed{args.seed}{shard}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(out_dir / "rows.parquet")
    except Exception as e:
        print(f"  (parquet skipped: {e})")

    def auc(pos, neg):
        pos = [x for x in pos if np.isfinite(x)]; neg = [x for x in neg if np.isfinite(x)]
        if not pos or not neg: return float("nan")
        a = np.array(pos + neg); o = np.argsort(a); r = np.empty(len(a)); r[o] = np.arange(len(a))
        s = a[o]; i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j+1] == s[i]: j += 1
            if j > i: r[o[i:j+1]] = (i + j) / 2
            i = j + 1
        return (r[:len(pos)].sum() - len(pos)*(len(pos)-1)/2) / (len(pos)*len(neg))

    def report(rs, label):
        H = [r for r in rs if r["hallucinated"]]; G = [r for r in rs if not r["hallucinated"]]
        if not H or not G:
            print(f"  {label}: 표본부족 (H={len(H)},G={len(G)})"); return
        print(f"\n  [{label}]  hall={len(H)} grounded={len(G)}")
        for k in ("M_c", "v_N_c", "maxphi_c"):
            a = auc([r[k] for r in H], [r[k] for r in G]); a = max(a, 1 - a)
            print(f"    AUC[{k:9s}]={a:.3f}  (hall {np.mean([r[k] for r in H]):+.2f} "
                  f"vs grnd {np.mean([r[k] for r in G]):+.2f})")
        base = len(H)/len(rs); allM = sorted(r["M_c"] for r in rs)
        thr = allM[len(allM)//4]; low = [r for r in rs if r["M_c"] <= thr]
        p = np.mean([r["hallucinated"] for r in low])
        print(f"    base P(hall)={base:.3f}  P(hall|M_c bottom25%)={p:.3f}  lift={p/base:.2f}x")

    print(f"\n=== OBJECT-LEVEL v_c (AMBER, {tag}) objs={len(rows)} skip_img={n_skip} ===")
    report(rows, "all objects")
    seen = defaultdict(set); first = []
    byimg = defaultdict(list)
    for r in rows: byimg[r["iid"]].append(r)
    for iid, os_ in byimg.items():
        s = set()
        for o in os_:
            if o["obj"] not in s:
                s.add(o["obj"]); first.append(o)
    report(first, "first-mention only")
    print(f"\nwrote {out_dir}/rows.jsonl  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
