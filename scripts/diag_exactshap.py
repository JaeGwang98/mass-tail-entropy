r"""ExactSHAP vs LOO diagnostic on POPE (binary-QA) and CHAIR (captioning).

For each sample it generates the model's actual output (greedy or sampling),
then attributes the realized span with BOTH estimators under the SAME union
mean-fill masking, plus prior baselines and cheap detection baselines.

Every row carries the PAPER-COMPARABLE entropy so the old LOO mass-tail
(draft Table 4/5) can be regenerated under both estimators (see
scripts/compare_loo_exact.py):

  H_loo   = _norm_entropy(softmax(phi_loo))     # the paper's exact SHAP-H
  H_exact = _norm_entropy(softmax(phi_exact))
  bin_loo / bin_exact in {over-conc, conc, mixed, spread, over-spread}

Labels:
  POPE  : HALL = false-yes (gt=no, pred=yes);   G+ = correct.
  CHAIR : HALL = caption with >=1 hallucinated object (CHAIR matcher);
          G+   = caption with 0 hallucinated objects.

Span attributed:
  POPE  : the generated yes/no answer.
  CHAIR : the first sentence of the generated caption (paper lookahead protocol);
          full caption text is used for object/hallucination labeling.

Examples
--------
  python -m scripts.diag_exactshap --bench pope  --setting random --limit 100 --decode greedy
  python -m scripts.diag_exactshap --bench chair --limit 100 --decode greedy
  python -m scripts.diag_exactshap --bench chair --limit 100 --decode sample-A --seed 1234
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.utils.common import PROJECT_ROOT, load_config, set_seed
from src.utils.segmentation import PanopticSegmenter
from src.benchmarks.pope import POPE_QUESTION_SUFFIX, parse_yes_no
from src.decoding.baseline import _extra_from_enc
from src.decoding.ours_msb_rolling import _is_sentence_end
from src.decoding.ours_sbc import _norm_entropy          # paper's exact SHAP-H
from src.decoding import exact_shap as ES

DECODE = {  # name -> (do_sample, temperature, top_p)
    "greedy":   (False, 0.0, 1.0),
    "sample-A": (True,  1.0, 1.0),   # VCD setting
    "sample-B": (True,  0.7, 0.9),
}
BIN_NAMES = ["over-conc", "conc", "mixed", "spread", "over-spread"]


def bin_of(H: float) -> str:
    """5 quintile bins, matching draft §3.1."""
    if H < 0.2:  return BIN_NAMES[0]
    if H < 0.4:  return BIN_NAMES[1]
    if H < 0.6:  return BIN_NAMES[2]
    if H < 0.8:  return BIN_NAMES[3]
    return BIN_NAMES[4]


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------
def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(dim=-1)
    remove = cum - probs > top_p
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, sorted_idx, sorted_logits)


@torch.no_grad()
def generate(wrapper, enc, do_sample: bool, temperature: float, top_p: float,
             max_steps: int, stop_at_sentence: bool
             ) -> Tuple[List[int], List[int], float]:
    """Generate output tokens.  Returns (full_tokens, first_sentence_tokens,
    first-token max-softmax).  ``first_sentence_tokens`` == full_tokens for the
    POPE (short) case; for CHAIR it is the span attributed by SHAP."""
    extra = _extra_from_enc(enc)
    out = wrapper.prefill(enc["input_ids"], enc["pixel_values"],
                          enc.get("attention_mask"), **extra)
    logits = out.logits[:, -1, :]
    pkv = out.past_key_values
    eos = int(wrapper.tokenizer.eos_token_id)
    maxsoftmax = float(torch.softmax(logits.float(), dim=-1).max().item())

    full: List[int] = []
    first_sent: List[int] = []
    sentence_done = False
    for step in range(max_steps):
        if do_sample:
            scaled = _top_p_filter(logits.float() / max(temperature, 1e-6), top_p)
            nxt = torch.multinomial(torch.softmax(scaled, dim=-1), 1)
        else:
            nxt = logits.argmax(-1, keepdim=True)
        tok = int(nxt.item())
        if tok == eos:
            break
        full.append(tok)
        if not sentence_done:
            first_sent.append(tok)
            if step + 1 >= 2 and _is_sentence_end(wrapper.tokenizer, tok):
                sentence_done = True
                if stop_at_sentence:
                    break
        logits, pkv = wrapper.decode_step(nxt, pkv)
        logits = logits[:, -1, :]
    return full, first_sent, maxsoftmax


# ---------------------------------------------------------------------------
# per-sample attribution -> row
# ---------------------------------------------------------------------------
def attribute(wrapper, segmenter, img, enc, span, m2f, max_batch) -> Optional[dict]:
    segments = segmenter.segment(img, min_area_frac=m2f["min_area_frac"],
                                 max_segments=m2f["max_segments"])
    res = ES.exact_shap_phis(wrapper, img, segments, enc["input_ids"], span,
                             max_batch=max_batch)
    if res is None:
        return None
    loo = ES.loo_phis(wrapper, img, segments, enc["input_ids"], span,
                      max_batch=max_batch)
    base = ES.prior_baselines(wrapper, enc, span)
    feats = ES.signature_features(res, b0=base["b0"], span_len=len(span))
    H_exact = _norm_entropy(res["phi"])
    H_loo = _norm_entropy(loo) if loo is not None else float("nan")
    row = {
        "K": int(res["phi"].shape[0]), "span_len": len(span),
        "phi_exact": res["phi"].tolist(),
        "phi_loo": (loo.tolist() if loo is not None else None),
        "M": res["M"], "M_per_tok": feats["M_per_tok"],
        "v_bg": res["v_bg"], "v_N": res["v_N"], "b0": base["b0"],
        "grounding_vs_b0": feats.get("grounding_vs_b0"),
        "grounding_vs_b0_per_tok": feats.get("grounding_vs_b0_per_tok"),
        "r_prior_bg": feats["r_prior_bg"], "r_prior_b0": feats.get("r_prior_b0"),
        "H_pos": feats["H_pos"], "top1_share": feats["top1_share"],
        "maxphi": feats["maxphi"], "minphi": feats["minphi"],
        # PAPER-COMPARABLE entropy + bins (LOO vs Exact)
        "H_loo": H_loo, "H_exact": H_exact,
        "bin_loo": (bin_of(H_loo) if loo is not None else None),
        "bin_exact": bin_of(H_exact),
        # cheap detection baselines
        "seq_logprob": res["v_N"], "seq_logprob_per_tok": res["v_N"]/len(span),
    }
    return row


# ---------------------------------------------------------------------------
# POPE driver
# ---------------------------------------------------------------------------
def run_pope(wrapper, segmenter, cfg, args):
    data_dir = PROJECT_ROOT / cfg["benchmarks"]["pope"]["data_dir"]
    image_dir = PROJECT_ROOT / cfg["benchmarks"]["pope"]["image_dir"]
    qs = []
    with open(data_dir / f"coco_pope_{args.setting}.json") as f:
        for line in f:
            if line.strip():
                qs.append(json.loads(line))
    qs = qs[:args.limit] if args.limit else qs
    do_sample, temp, top_p = DECODE[args.decode]
    m2f = cfg["mask2former"]
    rows, img_cache, t0, n_skip = [], {}, time.time(), 0
    for i, q in enumerate(qs):
        try:
            img = img_cache.get(q["image"])
            if img is None:
                img = Image.open(image_dir / q["image"]).convert("RGB")
                img_cache[q["image"]] = img
            enc = wrapper.prepare_inputs(img, q["text"] + POPE_QUESTION_SUFFIX)
            full, _, maxsoftmax = generate(wrapper, enc, do_sample, temp, top_p,
                                           max_steps=24, stop_at_sentence=True)
            if not full:
                n_skip += 1; continue
            row = attribute(wrapper, segmenter, img, enc, full, m2f, args.max_batch)
            if row is None:
                n_skip += 1; continue
            text = wrapper.tokenizer.decode(full, skip_special_tokens=True)
            pred, gt = parse_yes_no(text), q["label"].lower()
            row.update({
                "bench": "pope", "qid": q["question_id"], "image": q["image"],
                "question": q["text"], "span_text": text,
                "gt": gt, "pred": pred, "correct": (pred == gt),
                "is_hall": (gt == "no" and pred == "yes"),
                "maxsoftmax": maxsoftmax, "decode": args.decode, "seed": args.seed,
            })
            rows.append(row)
        except Exception as e:
            n_skip += 1
            print(f"  [skip qid={q.get('question_id')}] {type(e).__name__}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(qs)} kept={len(rows)} skip={n_skip} "
                  f"({(time.time()-t0)/(i+1):.1f}s/q)", flush=True)
            gc.collect(); torch.cuda.empty_cache()
    return rows, n_skip


# ---------------------------------------------------------------------------
# MME driver (hallucination subset; binary-QA, like POPE but separate benchmark)
# ---------------------------------------------------------------------------
def run_mme(wrapper, segmenter, cfg, args):
    from src.benchmarks.mme import load_mme_subset, _to_pil
    df = load_mme_subset()
    if args.limit:
        df = df.iloc[:args.limit].reset_index(drop=True)
    do_sample, temp, top_p = DECODE[args.decode]
    m2f = cfg["mask2former"]
    rows, t0, n_skip = [], time.time(), 0
    for i, q in df.iterrows():
        try:
            img = _to_pil(q["image"])
            enc = wrapper.prepare_inputs(img, str(q["question"]))  # already yes/no
            full, _, maxsoftmax = generate(wrapper, enc, do_sample, temp, top_p,
                                           max_steps=24, stop_at_sentence=True)
            if not full:
                n_skip += 1; continue
            row = attribute(wrapper, segmenter, img, enc, full, m2f, args.max_batch)
            if row is None:
                n_skip += 1; continue
            text = wrapper.tokenizer.decode(full, skip_special_tokens=True)
            pred, gt = parse_yes_no(text), str(q["answer"]).strip().lower()
            row.update({
                "bench": "mme", "qid": str(q["question_id"]),
                "category": str(q["category"]),
                "question": str(q["question"]), "span_text": text,
                "gt": gt, "pred": pred, "correct": (pred == gt),
                "is_hall": (gt == "no" and pred == "yes"),
                "is_wrong": (pred != gt),
                "maxsoftmax": maxsoftmax, "decode": args.decode, "seed": args.seed,
            })
            rows.append(row)
        except Exception as e:
            n_skip += 1
            print(f"  [skip qid={q.get('question_id')}] {type(e).__name__}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(df)} kept={len(rows)} skip={n_skip} "
                  f"({(time.time()-t0)/(i+1):.1f}s/q)", flush=True)
            gc.collect(); torch.cuda.empty_cache()
    return rows, n_skip


# ---------------------------------------------------------------------------
# CHAIR driver
# ---------------------------------------------------------------------------
def run_chair(wrapper, segmenter, cfg, args):
    from src.benchmarks.chair import (CHAIR, CHAIR_PROMPT, sample_image_ids)
    cc = cfg["benchmarks"]["chair"]
    image_dir = PROJECT_ROOT / cc["image_dir"]
    ann_dir = PROJECT_ROOT / cc["annotations_dir"]
    n = args.limit or cc["n_images"]
    chair = CHAIR(PROJECT_ROOT / cc["synonyms_file"])
    sampled = sample_image_ids(image_dir, n, cc["seed"])
    image_ids = [iid for iid, _ in sampled]
    print(f"building CHAIR GT for {len(image_ids)} images ...")
    gt = chair.build_gt(ann_dir / "instances_val2014.json",
                        ann_dir / "captions_val2014.json", image_ids)

    do_sample, temp, top_p = DECODE[args.decode]
    m2f = cfg["mask2former"]
    rows, t0, n_skip = [], time.time(), 0
    for i, (iid, fname) in enumerate(sampled):
        try:
            img = Image.open(image_dir / fname).convert("RGB")
            enc = wrapper.prepare_inputs(img, CHAIR_PROMPT)
            # generate full caption (capped), attribute its FIRST SENTENCE
            full, first_sent, maxsoftmax = generate(
                wrapper, enc, do_sample, temp, top_p,
                max_steps=args.cap_tokens, stop_at_sentence=False)
            span = first_sent if first_sent else full
            if not span:
                n_skip += 1; continue
            row = attribute(wrapper, segmenter, img, enc, span, m2f, args.max_batch)
            if row is None:
                n_skip += 1; continue
            caption = wrapper.tokenizer.decode(full, skip_special_tokens=True)
            _, node_words = chair.caption_to_objects(caption)
            gt_set = gt[iid]
            hallu = [w for w in node_words if w not in gt_set]
            row.update({
                "bench": "chair", "qid": iid, "image": fname,
                "question": CHAIR_PROMPT,
                "span_text": wrapper.tokenizer.decode(span, skip_special_tokens=True),
                "caption": caption,
                "gt_objects": sorted(gt_set), "gen_objects": node_words,
                "hallucinated_objects": hallu, "n_hallu": len(hallu),
                "n_obj": len(node_words),
                # CHAIR labels: HALL = >=1 hallucinated object; G+ = grounded (0)
                "is_hall": (len(hallu) >= 1),
                "correct": (len(node_words) > 0 and len(hallu) == 0),
                "maxsoftmax": maxsoftmax, "decode": args.decode, "seed": args.seed,
            })
            rows.append(row)
        except Exception as e:
            n_skip += 1
            print(f"  [skip iid={iid}] {type(e).__name__}: {e}")
        if (i + 1) % 20 == 0:
            ch = sum(r["is_hall"] for r in rows)
            print(f"  {i+1}/{len(sampled)} kept={len(rows)} HALL={ch} "
                  f"skip={n_skip} ({(time.time()-t0)/(i+1):.1f}s/img)", flush=True)
            gc.collect(); torch.cuda.empty_cache()
    return rows, n_skip


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="pope", choices=["pope", "chair", "mme"])
    ap.add_argument("--setting", default="random",
                    choices=["random", "popular", "adversarial"],
                    help="POPE split (ignored for CHAIR).")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--decode", default="greedy", choices=list(DECODE))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cap-tokens", type=int, default=128,
                    help="CHAIR: max caption tokens to generate (cost bound).")
    ap.add_argument("--model", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--out-dir", default="results/exactshap")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model

    from src.models import build_wrapper
    print(f"loading {cfg['model']['name']} ...", flush=True)
    wrapper = build_wrapper(model_name=cfg["model"]["name"],
                            dtype=getattr(torch, cfg["model"]["dtype"]),
                            attn_implementation=cfg["model"]["attn_implementation"])
    print("loading Mask2Former ...", flush=True)
    segmenter = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                                  dtype=getattr(torch, cfg["model"]["dtype"]))
    set_seed(args.seed)

    if args.bench == "pope":
        rows, n_skip = run_pope(wrapper, segmenter, cfg, args)
        tag = f"pope-{args.setting}"
    elif args.bench == "mme":
        rows, n_skip = run_mme(wrapper, segmenter, cfg, args)
        tag = "mme"
    else:
        rows, n_skip = run_chair(wrapper, segmenter, cfg, args)
        tag = "chair"

    model_tag = cfg["model"]["name"].split("/")[-1]
    out_dir = (PROJECT_ROOT / args.out_dir /
               f"{model_tag}_{tag}_{args.decode}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(out_dir / "rows.parquet")
    except Exception as e:
        print(f"  (parquet skipped: {e})")

    # quick sanity summary (NOT the formal analysis)
    if rows:
        def gmean(key, pred):
            xs = [r[key] for r in rows if pred(r) and r.get(key) is not None
                  and np.isfinite(r.get(key, float("nan")))]
            return (len(xs), float(np.mean(xs)) if xs else float("nan"))
        nh = sum(r["is_hall"] for r in rows)
        print(f"\n=== {model_tag} {tag} {args.decode} seed{args.seed} ===")
        print(f"kept={len(rows)} skipped={n_skip} HALL={nh} "
              f"G+={sum(r['correct'] for r in rows)}")
        for key in ("M", "grounding_vs_b0", "H_loo", "H_exact"):
            ng, mg = gmean(key, lambda r: r["correct"])
            nhh, mh = gmean(key, lambda r: r["is_hall"])
            print(f"  {key:18s} G+(n={ng}) {mg:+.3f} | HALL(n={nhh}) {mh:+.3f}")
        # paper-style dominant tail (LOO vs Exact)
        for est in ("bin_loo", "bin_exact"):
            from collections import Counter
            c = Counter(r[est] for r in rows if r.get(est))
            dom = c.most_common(1)[0] if c else ("-", 0)
            print(f"  {est}: dominant={dom[0]} ({dom[1]}/{sum(c.values())})  {dict(c)}")
    print(f"\nwrote {out_dir}/rows.jsonl")


if __name__ == "__main__":
    main()
