"""POPE evaluator (Li et al., EMNLP 2023; replicating the VCD protocol).

Reads the JSONL files under ``data/POPE/`` (one line per question), generates
an answer with the chosen decoding method, parses the first yes/no token,
and reports Accuracy / Precision / Recall / F1 per (random/popular/adversarial)
split.  The prompt mirrors the VCD official repo:

    "Please answer the following question with yes or no. {question}"

Multiple sampling runs (``--n_runs``) are averaged, mirroring Tab. 1 of VCD.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

from ..models.llava_wrapper import LlavaWrapper
from ..utils.common import PROJECT_ROOT, load_config, set_seed
from ..decoding.baseline import sample_decode, greedy_decode
from ..decoding.vcd import vcd_decode
from ..decoding.aif import aif_decode
# (legacy ours_v3 — lazy-imported inside make_decoder when method=='ours')

# Official VCD/OPERA/LURE POPE prompt: append the suffix that constrains
# LLaVA-1.5 to a one-word answer.  Without it, our baseline yes-ratio sits at
# ~55% (matches HF + the original liuhaotian checkpoint), but VCD's paper
# baseline yes-ratio is ~40%.  Adding the suffix reproduces the paper's
# yes-ratio (~39.5) and recovers the +5 acc gain VCD reports.
POPE_QUESTION_SUFFIX = " Please answer this question with one word."
YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
NO_RE = re.compile(r"\bno\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# parsing & metric utilities
# ---------------------------------------------------------------------------
def parse_yes_no(text: str) -> str:
    text = text.strip()
    if YES_RE.search(text):
        return "yes"
    if NO_RE.search(text):
        return "no"
    # Fallback: try first whitespace-separated token
    head = text.split()[0].lower() if text.split() else ""
    return "yes" if head.startswith("y") else "no"


def compute_metrics(preds: Sequence[str], labels: Sequence[str]) -> Dict[str, float]:
    tp = sum(1 for p, l in zip(preds, labels) if p == "yes" and l == "yes")
    tn = sum(1 for p, l in zip(preds, labels) if p == "no" and l == "no")
    fp = sum(1 for p, l in zip(preds, labels) if p == "yes" and l == "no")
    fn = sum(1 for p, l in zip(preds, labels) if p == "no" and l == "yes")
    n = tp + tn + fp + fn
    accuracy = (tp + tn) / max(1, n)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    yes_ratio = (tp + fp) / max(1, n)
    return dict(accuracy=accuracy, precision=precision, recall=recall,
                f1=f1, yes_ratio=yes_ratio,
                tp=tp, tn=tn, fp=fp, fn=fn, n=n)


# ---------------------------------------------------------------------------
# Decoder dispatcher
# ---------------------------------------------------------------------------
# Early prototypes whose decoder modules are not part of this release.
_UNRELEASED_METHODS = ("ours_maskk", "ours_penalty", "ours_penalty_sent",
                       "ours_combined", "ssl")

def make_decoder(method: str, wrapper: LlavaWrapper, cfg: dict,
                 segmenter=None) -> Callable:
    vc = cfg["vcd"]
    ours = cfg["ours"]
    aifc = cfg["aif"]
    max_new = cfg["benchmarks"]["pope"]["max_new_tokens"]

    if method in _UNRELEASED_METHODS:
        raise ValueError(f"method {method!r} is an early prototype that is not "
                         "part of the paper and is not included in this release")

    if method == "baseline":
        return lambda img, q: sample_decode(wrapper, img, q,
                                            max_new_tokens=max_new)
    if method == "baseline_greedy":
        return lambda img, q: greedy_decode(wrapper, img, q,
                                            max_new_tokens=max_new)
    if method == "vcd":
        return lambda img, q: vcd_decode(wrapper, img, q,
                                         max_new_tokens=max_new,
                                         alpha=vc["alpha"], beta=vc["beta"],
                                         noise_step=vc["noise_steps_pope"],
                                         sampling="direct")
    if method == "vcd_greedy":
        return lambda img, q: vcd_decode(wrapper, img, q,
                                         max_new_tokens=max_new,
                                         alpha=vc["alpha"], beta=vc["beta"],
                                         noise_step=vc["noise_steps_pope"],
                                         sampling="greedy")
    if method == "aif":
        return lambda img, q: aif_decode(wrapper, img, q,
                                         max_new_tokens=max_new,
                                         sampling="greedy",
                                         mask_ratios=tuple(aifc["mask_ratios"]),
                                         max_ratio=aifc.get("max_ratio", 0.5))
    if method == "ours_pmi":
        from ..decoding.ours_pmi import ours_pmi_decode
        return lambda img, q: ours_pmi_decode(wrapper, img, q,
                                              max_new_tokens=max_new,
                                              alpha=ours.get("pmi_alpha", 1.0),
                                              beta=ours.get("beta", 0.1),
                                              sampling="greedy")
    if method == "ours_maskk":
        from ..decoding.ours_maskk import ours_maskk_decode
        if segmenter is None:
            raise ValueError("ours_maskk requires a PanopticSegmenter")
        return lambda img, q: ours_maskk_decode(wrapper, segmenter, img, q,
                                                max_new_tokens=max_new,
                                                alpha=ours.get("pmi_alpha", 1.0),
                                                beta=ours.get("beta", 0.1),
                                                top_k=ours.get("maskk_top_k", 2),
                                                sampling="greedy")
    if method in ("ours_sbc", "ours_sbc_v2", "ours_pmi_guard",
                  "ours_no_h", "ours_logit_h", "ours_attn_h",
                  "ours_task_pmi", "ours_task_msb", "ours_lazy", "ours_lazy_attn"):
        from ..decoding.ours_sbc import ours_sbc_decode
        if segmenter is None:
            raise ValueError(f"{method} requires a PanopticSegmenter")
        if method == "ours_sbc_v2": gv = "v2"
        elif method == "ours_pmi_guard": gv = "pmi_guard_only"
        elif method == "ours_lazy": gv = "lazy"
        elif method == "ours_lazy_attn": gv = "lazy_attn"
        elif method == "ours_no_h": gv = "no_h"
        elif method == "ours_logit_h": gv = "logit_h"
        elif method == "ours_attn_h": gv = "attn_h"
        elif method == "ours_task_pmi": gv = "task_pmi"
        elif method == "ours_task_msb": gv = "task_msb"
        else: gv = "v3"
        return lambda img, q: ours_sbc_decode(
            wrapper, segmenter, img, q,
            max_new_tokens=max_new,
            boost_factor=ours.get("msb_boost_factor", 1.8),
            top_k=ours.get("msb_top_k", 2),
            pmi_alpha=ours.get("pmi_alpha", 1.0), beta=ours.get("beta", 0.1),
            gate_version=gv,
            tau_mid=ours.get("sbc_tau_mid", 0.5),
            tau_lo=ours.get("sbc_tau_lo", 0.25), tau_hi=ours.get("sbc_tau_hi", 0.75),
            image_margin_thresh=ours.get("image_margin_thresh", 0.5),
            max_segments=ours.get("max_segments", 6),
            return_route=True)
    if method == "ours":
        from ..decoding.ours_v3 import ours_v3_decode
        if segmenter is None:
            raise ValueError("ours requires a PanopticSegmenter")
        return lambda img, q: ours_v3_decode(wrapper, segmenter, img, q,
                                             max_new_tokens=max_new,
                                             alpha=ours["alpha"],
                                             beta=ours["beta"],
                                             lookahead=ours["lookahead"],
                                             sampling="greedy")
    if method == "ours_v4":
        from ..decoding.ours_v4 import ours_v4_decode
        if segmenter is None:
            raise ValueError("ours_v4 requires a PanopticSegmenter")
        boost = ours.get("v4_boost_factor", 2.0)
        return lambda img, q: ours_v4_decode(wrapper, segmenter, img, q,
                                             max_new_tokens=max_new,
                                             boost_factor=boost,
                                             lookahead=ours["lookahead"])
    if method == "ours_msb":
        from ..decoding.ours_msb import ours_msb_decode
        if segmenter is None:
            raise ValueError("ours_msb requires a PanopticSegmenter")
        boost = ours.get("msb_boost_factor", 1.8)
        topk = ours.get("msb_top_k", 2)
        return lambda img, q: ours_msb_decode(wrapper, segmenter, img, q,
                                              max_new_tokens=max_new,
                                              boost_factor=boost, top_k=topk,
                                              lookahead=ours["lookahead"])
    if method == "ours_msb_rolling":
        from ..decoding.ours_msb_rolling import ours_msb_rolling_decode
        if segmenter is None:
            raise ValueError("ours_msb_rolling requires a PanopticSegmenter")
        boost = ours.get("msb_boost_factor", 1.8)
        topk = ours.get("msb_top_k", 2)
        max_re = ours.get("msb_max_remeasures", 2)
        return lambda img, q: ours_msb_rolling_decode(
            wrapper, segmenter, img, q,
            max_new_tokens=max_new,
            boost_factor=boost, top_k=topk,
            lookahead=ours["lookahead"],
            max_remeasures=max_re)
    if method == "ours_penalty":
        from ..decoding.ours_penalty import ours_penalty_decode
        if segmenter is None:
            raise ValueError("ours_penalty requires a PanopticSegmenter")
        pen = ours.get("penalty_factor", 0.5)
        gate = ours.get("penalty_gate_threshold", 0.05)
        return lambda img, q: ours_penalty_decode(
            wrapper, segmenter, img, q,
            max_new_tokens=max_new,
            penalty_factor=pen, gate_threshold=gate,
            lookahead=ours["lookahead"])
    if method == "ours_msb_sent":
        from ..decoding.ours_msb import ours_msb_decode
        if segmenter is None:
            raise ValueError("ours_msb_sent requires a PanopticSegmenter")
        boost = ours.get("msb_boost_factor", 1.8)
        topk = ours.get("msb_top_k", 2)
        return lambda img, q: ours_msb_decode(
            wrapper, segmenter, img, q,
            max_new_tokens=max_new,
            boost_factor=boost, top_k=topk,
            lookahead=ours["lookahead"],
            use_sentence_lookahead=True)
    if method == "ours_penalty_sent":
        from ..decoding.ours_penalty import ours_penalty_decode
        if segmenter is None:
            raise ValueError("ours_penalty_sent requires a PanopticSegmenter")
        pen = ours.get("penalty_factor", 0.5)
        gate = ours.get("penalty_gate_threshold", 0.05)
        return lambda img, q: ours_penalty_decode(
            wrapper, segmenter, img, q,
            max_new_tokens=max_new,
            penalty_factor=pen, gate_threshold=gate,
            lookahead=ours["lookahead"],
            use_sentence_lookahead=True)
    if method == "ours_combined":
        from ..decoding.ours_combined import ours_combined_decode
        if segmenter is None:
            raise ValueError("ours_combined requires a PanopticSegmenter")
        boost = ours.get("msb_boost_factor", 1.8)
        pen = ours.get("penalty_factor", 0.5)
        gate = ours.get("penalty_gate_threshold", 0.05)
        topk = ours.get("msb_top_k", 2)
        return lambda img, q: ours_combined_decode(
            wrapper, segmenter, img, q,
            max_new_tokens=max_new,
            boost_factor=boost, penalty_factor=pen,
            gate_threshold=gate, top_k=topk)
    if method == "opera":
        from ..decoding.opera import opera_greedy_decode
        op = cfg.get("opera", {})
        return lambda img, q: opera_greedy_decode(
            wrapper, img, q,
            max_new_tokens=max_new,
            alpha=op.get("alpha", 1.0),
            sigma=op.get("sigma", 50.0),
            k_window=op.get("k_window", 8),
            ncan=op.get("ncan", 5))
    if method == "pai":
        from ..decoding.pai import pai_decode
        pai_cfg = cfg.get("pai", {})
        return lambda img, q: pai_decode(
            wrapper, img, q,
            max_new_tokens=max_new,
            alpha=pai_cfg.get("alpha", 0.5),
            gamma_cfg=pai_cfg.get("gamma_cfg", 1.1),
            use_cfg=pai_cfg.get("use_cfg", True),
            start_layer=pai_cfg.get("start_layer", 2),
            end_layer=pai_cfg.get("end_layer", 32))
    if method == "ssl":
        from ..decoding.ssl import ssl_decode
        ssl_cfg = cfg.get("ssl", {})
        return lambda img, q: ssl_decode(
            wrapper, img, q,
            max_new_tokens=max_new,
            gamma=ssl_cfg.get("gamma", 0.8),
            layer=ssl_cfg.get("layer", 30),
            sae_path=ssl_cfg.get("sae_path",
                "repro/SSL/data/sae"
                "/llama3-llava-next-8b-hf-sae-131k/model.layers.24"))
    raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------------
# main eval loop
# ---------------------------------------------------------------------------
@dataclass
class POPEResult:
    method: str
    setting: str
    n_runs: int
    metrics_per_run: List[Dict[str, float]] = field(default_factory=list)
    aggregate: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_questions: int = 0
    seconds: float = 0.0


def _aggregate(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = ["accuracy", "precision", "recall", "f1", "yes_ratio"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=np.float64)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return out


def evaluate_setting(method: str, setting: str, wrapper: LlavaWrapper,
                     cfg: dict, segmenter=None,
                     limit: int | None = None,
                     n_runs: int | None = None,
                     out_dir: Path | None = None) -> POPEResult:
    pope_cfg = cfg["benchmarks"]["pope"]
    data_dir = PROJECT_ROOT / pope_cfg["data_dir"]
    image_dir = PROJECT_ROOT / pope_cfg["image_dir"]
    n_runs = n_runs if n_runs is not None else pope_cfg["n_runs"]
    json_path = data_dir / f"coco_pope_{setting}.json"

    questions = []
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if limit is not None:
        questions = questions[:limit]

    # Sampling-based decoders are run n_runs times; deterministic ones once.
    deterministic = method in ("aif", "ours", "ours_v4", "ours_msb",
                               "ours_msb_rolling", "ours_penalty",
                               "ours_msb_sent", "ours_penalty_sent",
                               "ours_combined", "ours_pmi", "ours_maskk", "ours_sbc", "ours_sbc_v2", "ours_pmi_guard",
                               "ours_no_h", "ours_logit_h", "ours_attn_h",
                               "ours_task_pmi", "ours_task_msb", "ours_lazy", "ours_lazy_attn",
                               "baseline_greedy", "vcd_greedy", "pai")
    runs = 1 if deterministic else n_runs
    decoder = make_decoder(method, wrapper, cfg, segmenter=segmenter)

    res = POPEResult(method=method, setting=setting, n_runs=runs,
                     n_questions=len(questions))
    t0 = time.time()
    image_cache: Dict[str, Image.Image] = {}

    for r in range(runs):
        set_seed(1234 + r)
        preds, labels, raw = [], [], []
        for i, q in enumerate(questions):
            img_name = q["image"]
            img = image_cache.get(img_name)
            if img is None:
                img = Image.open(image_dir / img_name).convert("RGB")
                image_cache[img_name] = img
            result = decoder(img, q["text"] + POPE_QUESTION_SUFFIX)
            if isinstance(result, tuple):
                text, route = result
            else:
                text, route = result, None
            ans = parse_yes_no(text)
            preds.append(ans)
            labels.append(q["label"].lower())
            raw.append({"qid": q["question_id"], "image": img_name,
                        "question": q["text"], "gt": q["label"],
                        "pred": ans, "raw_text": text, "route": route})
            if (i + 1) % 200 == 0:
                print(f"    [{method}/{setting}/run{r}] {i+1}/{len(questions)}")
            if i % 20 == 0:
                gc.collect(); torch.cuda.empty_cache()
        m = compute_metrics(preds, labels)
        res.metrics_per_run.append(m)
        if out_dir is not None:
            (out_dir / f"raw_{method}_{setting}_run{r}.jsonl").write_text(
                "\n".join(json.dumps(x) for x in raw))
    res.seconds = time.time() - t0
    res.aggregate = _aggregate(res.metrics_per_run)
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["baseline", "baseline_greedy", "vcd",
                             "vcd_greedy", "aif", "opera", "pai", "ssl",
                             "ours_pmi", "ours_msb_sent", "ours_sbc",
                             "ours_sbc_v2", "ours_pmi_guard",
                             "ours_no_h", "ours_logit_h", "ours_attn_h",
                             "ours_task_pmi", "ours_task_msb", "ours_lazy", "ours_lazy_attn"])
    ap.add_argument("--setting", default="all",
                    choices=["random", "popular", "adversarial", "all"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of questions per setting (debug).")
    ap.add_argument("--n-runs", type=int, default=None,
                    help="Override n_runs from config.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="results/pope")
    ap.add_argument("--tau-mid", type=float, default=None,
                    help="Override SBC v3 gate threshold (ablation/sweep).")
    ap.add_argument("--model", default=None,
                    help="Override model id (e.g. Qwen/Qwen2-VL-7B-Instruct).")
    ap.add_argument("--load-8bit", action="store_true",
                    help="Load the model in 8-bit (for 13B on a 24GB GPU).")
    ap.add_argument("--device-map", default=None,
                    help="HF device_map for fp16 multi-GPU split (e.g., 'auto').")
    ap.add_argument("--boost-factor", type=float, default=None,
                    help="Override SBC/MSB boost_factor (paper default 1.8, from configs/default.yaml).")
    ap.add_argument("--image-margin-thresh", type=float, default=None,
                    help="SBC v2 image-margin guard: skip PMI when image-conditioned top-token margin exceeds this (0=off).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.tau_mid is not None:
        cfg.setdefault("ours", {})["sbc_tau_mid"] = args.tau_mid
        print(f"[override] sbc_tau_mid = {args.tau_mid}")
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
        attn_implementation=cfg["model"]["attn_implementation"],
        load_in_8bit=args.load_8bit,
        device_map=args.device_map)
    segmenter = None
    if args.method in ("ours_sbc", "ours_sbc_v2", "ours_msb_sent", "ours_pmi_guard",
                       "ours_no_h", "ours_logit_h", "ours_attn_h",
                       "ours_task_pmi", "ours_task_msb", "ours_lazy", "ours_lazy_attn"):  # paper: SBC + MSB-only + App K ablations
        from ..utils.segmentation import PanopticSegmenter
        print(f"loading Mask2Former...")
        segmenter = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                                      dtype=getattr(torch, cfg["model"]["dtype"]))

    settings = (cfg["benchmarks"]["pope"]["settings"]
                if args.setting == "all" else [args.setting])
    summary = {}
    for s in settings:
        print(f"\n=== {args.method} on POPE/{s} ===")
        res = evaluate_setting(args.method, s, wrapper, cfg,
                               segmenter=segmenter, limit=args.limit,
                               n_runs=args.n_runs, out_dir=out_dir)
        summary[s] = {
            "n_questions": res.n_questions,
            "n_runs": res.n_runs,
            "seconds": res.seconds,
            "aggregate": res.aggregate,
            "per_run": res.metrics_per_run,
        }
        agg = res.aggregate
        print(f"  acc={agg['accuracy']['mean']*100:5.2f}±{agg['accuracy']['std']*100:.2f}  "
              f"prec={agg['precision']['mean']*100:5.2f}  "
              f"rec={agg['recall']['mean']*100:5.2f}  "
              f"f1={agg['f1']['mean']*100:5.2f}  "
              f"yes={agg['yes_ratio']['mean']*100:5.2f}")

    out_file = out_dir / f"summary_{args.method}.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_file}")


if __name__ == "__main__":
    main()
