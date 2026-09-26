r"""Object-level targeted v_c diagnostic on CHAIR (LLaVA).

The sentence-level pilots showed CHAIR hallucination is invisible at the caption
level (AUC~0.5) because correct objects swamp the one fake object.  This script
tests the user's ORIGINAL claim at the right granularity:

  "A hallucinated object has little/no visual support."

For each generated caption:
  1. generate caption (greedy), segment image (K<=6 panoptic segments).
  2. localize every COCO object word -> its caption token position(s) (char-offset
     bridge), label each hallucinated (not in GT) vs grounded (in GT).
  3. run 2^K coalition forwards over (prompt + full caption) ONCE; from the
     per-token log-probs extract, FOR EACH OBJECT, the targeted value
        v_c(S) = sum_{t in object tokens} log p(token_t | prefix, masked image S)
     and the exact Shapley phi_c, M_c = v_c(N)-v_c(bg), maxphi_c.
  4. dump ONE ROW PER OBJECT (well-powered: ~hundreds of objects).

Then analyze (object level):
  - AUC(M_c / maxphi_c : hallucinated vs grounded)
  - P(hallucinated | low M_c) and lift vs base rate  (the right metric, §critical)

Usage:
  python -m scripts.diag_vc_chair --limit 150 --cap-tokens 64
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.utils.common import PROJECT_ROOT, load_config, set_seed
from src.utils.segmentation import PanopticSegmenter, mask_image_with_segments
from src.decoding import exact_shap as ES


@torch.no_grad()
def greedy_caption(wrapper, enc, max_steps):
    out = wrapper.prefill(enc["input_ids"], enc["pixel_values"],
                          enc.get("attention_mask"))
    logits = out.logits[:, -1, :]; pkv = out.past_key_values
    eos = int(wrapper.tokenizer.eos_token_id)
    toks = []
    for _ in range(max_steps):
        nxt = logits.argmax(-1, keepdim=True); t = int(nxt.item())
        if t == eos: break
        toks.append(t)
        logits, pkv = wrapper.decode_step(nxt, pkv); logits = logits[:, -1, :]
    return toks


def token_char_offsets(tokenizer, ids):
    """(c_start, c_end) per token of ``ids`` via incremental decode over the
    decoded caption string (robust to subword splits)."""
    offs = []
    prev = ""
    for j in range(len(ids)):
        cur = tokenizer.decode(ids[:j + 1], skip_special_tokens=True)
        offs.append((len(prev), len(cur)))
        prev = cur
    return offs, prev  # prev == full decoded caption


def localize_objects(tokenizer, caption_ids, matcher, gt_set):
    """Return list of dict(obj, token_idx[list], hallucinated). Uses a char-span
    bridge: find COCO object surface words in the caption, map their char span to
    the generated token indices that overlap it."""
    offs, text = token_char_offsets(tokenizer, caption_ids)
    low = text.lower()
    words = [(m.group(), m.start(), m.end())
             for m in re.finditer(r"[a-zA-Z]+", low)]
    sing = matcher.singularize
    inv = matcher.inverse_synonym
    dwd = matcher.double_word_dict
    found = []
    i = 0
    while i < len(words):
        w, s, e = words[i]
        canon = None; span = (s, e)
        # double-word (e.g. "traffic light")
        if i + 1 < len(words):
            two = sing(w) + " " + sing(words[i + 1][0])
            if two in dwd and dwd[two] in inv:
                canon = inv[dwd[two]]; span = (s, words[i + 1][2]); i += 1
        if canon is None:
            sw = sing(w)
            if sw in inv:
                canon = inv[sw]
        if canon is not None:
            tok_idx = [j for j, (cs, ce) in enumerate(offs)
                       if cs < span[1] and ce > span[0]]
            if tok_idx:
                found.append({"obj": canon, "token_idx": tok_idx,
                              "hallucinated": canon not in gt_set})
        i += 1
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip the first N sampled images (for GPU sharding).")
    ap.add_argument("--cap-tokens", type=int, default=64)
    ap.add_argument("--fill", default="mean",
                    choices=["mean", "zero", "gaussian", "blur"],
                    help="Occlusion fill for v(S) (fill-robustness study).")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--model", default=None)
    ap.add_argument("--config", default=None)
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

    from src.benchmarks.chair import CHAIR, CHAIR_PROMPT, sample_image_ids
    cc = cfg["benchmarks"]["chair"]
    image_dir = PROJECT_ROOT / cc["image_dir"]
    ann = PROJECT_ROOT / cc["annotations_dir"]
    matcher = CHAIR(PROJECT_ROOT / cc["synonyms_file"])
    sampled = sample_image_ids(image_dir, args.offset + args.limit, cc["seed"])
    sampled = sampled[args.offset:]          # disjoint shard for GPU split
    gt = matcher.build_gt(ann / "instances_val2014.json",
                          ann / "captions_val2014.json",
                          [iid for iid, _ in sampled])
    set_seed(args.seed)

    rows = []
    t0 = time.time(); n_skip = 0
    for i, (iid, fname) in enumerate(sampled):
        try:
            img = Image.open(image_dir / fname).convert("RGB")
            enc = wrapper.prepare_inputs(img, CHAIR_PROMPT)
            cap_ids = greedy_caption(wrapper, enc, args.cap_tokens)
            if not cap_ids:
                n_skip += 1; continue
            objs = localize_objects(wrapper.tokenizer, cap_ids, matcher, gt[iid])
            if not objs:
                continue
            segs = seg.segment(img, min_area_frac=m2f["min_area_frac"],
                               max_segments=m2f["max_segments"])
            K = len(segs)
            if K < 2:
                n_skip += 1; continue
            subsets = ES.all_subsets(K)
            imgs = [mask_image_with_segments(img,
                        [segs[k].mask for k in range(K) if k not in S],
                        fill=args.fill)
                    for S in subsets]
            # (2^K, L) per-token logp from ONE pass over prompt+caption
            mat = ES.coalition_pertoken_logp(wrapper, imgs, enc["input_ids"],
                                             cap_ids, max_batch=args.max_batch)
            caption = wrapper.tokenizer.decode(cap_ids, skip_special_tokens=True)
            for o in objs:
                pos = o["token_idx"]
                vals = mat[:, pos].sum(axis=1)          # v_c(S) per coalition
                phi, v_bg, v_N, M = ES.exact_shap_from_coalition_values(
                    vals, subsets, K)
                phi_loo = ES.loo_from_coalition_values(vals, subsets, K)
                rows.append({
                    "iid": iid, "image": fname, "obj": o["obj"],
                    "hallucinated": bool(o["hallucinated"]),
                    "n_tok": len(pos), "K": K,
                    "M_c": float(M), "M_c_per_tok": float(M / len(pos)),
                    "v_bg_c": float(v_bg), "v_N_c": float(v_N),
                    "maxphi_c": float(phi.max()), "minphi_c": float(phi.min()),
                    # LOO (paper method) from the SAME coalition matrix (free)
                    "M_c_loo": float(phi_loo.sum()),
                    "maxphi_c_loo": float(phi_loo.max()),
                    "caption": caption,
                })
        except Exception as e:
            n_skip += 1
            print(f"  [skip iid={iid}] {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            nh = sum(r["hallucinated"] for r in rows)
            print(f"  {i+1}/{len(sampled)} objs={len(rows)} hallObj={nh} "
                  f"skip={n_skip} ({(time.time()-t0)/(i+1):.1f}s/img)", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    tag = cfg["model"]["name"].split("/")[-1]
    shard = "" if args.offset == 0 else f"_off{args.offset}"
    fillt = "" if args.fill == "mean" else f"_{args.fill}"
    out_dir = (PROJECT_ROOT / args.out_dir /
               f"{tag}_chair_vc_greedy_seed{args.seed}{fillt}{shard}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(out_dir / "rows.parquet")
    except Exception as e:
        print(f"  (parquet skipped: {e})")

    # ---- object-level analysis: the decisive test ----
    H = [r for r in rows if r["hallucinated"]]
    G = [r for r in rows if not r["hallucinated"]]
    def auc(pos, neg):
        pos = [x for x in pos if np.isfinite(x)]; neg = [x for x in neg if np.isfinite(x)]
        if not pos or not neg: return float("nan")
        a = np.array(pos + neg); o = np.argsort(a); r = np.empty(len(a)); r[o] = np.arange(len(a))
        # tie-avg
        s = a[o]; i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j+1] == s[i]: j += 1
            if j > i: r[o[i:j+1]] = (i + j) / 2
            i = j + 1
        U = r[:len(pos)].sum() - len(pos)*(len(pos)-1)/2
        return U/(len(pos)*len(neg))
    print(f"\n=== OBJECT-LEVEL v_c (CHAIR, {tag}) ===")
    print(f"objects={len(rows)}  hallucinated={len(H)}  grounded={len(G)}  "
          f"skip_img={n_skip}")
    if H and G:
        for key in ("M_c", "M_c_per_tok", "maxphi_c", "v_N_c"):
            a = auc([r[key] for r in H], [r[key] for r in G])
            a = max(a, 1 - a) if a == a else a
            print(f"  AUC[{key:12s}] hall vs grounded = {a:.3f}   "
                  f"(hall mean={np.mean([r[key] for r in H]):+.2f}  "
                  f"grnd mean={np.mean([r[key] for r in G]):+.2f})")
        # P(hall | low signal) + lift  (the right metric)
        base = len(H) / len(rows)
        allM = sorted(r["M_c"] for r in rows)
        thr = allM[len(allM)//4]            # bottom-quartile M_c threshold
        low = [r for r in rows if r["M_c"] <= thr]
        p_low = np.mean([r["hallucinated"] for r in low]) if low else float("nan")
        print(f"\n  base P(hall)={base:.3f}")
        print(f"  P(hall | M_c in bottom 25%)={p_low:.3f}  lift={p_low/base:.2f}x "
              f"(>1 => low visual support predicts hallucination)")
    print(f"\nwrote {out_dir}/rows.jsonl  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
