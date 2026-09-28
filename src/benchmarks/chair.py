"""CHAIR evaluator (Rohrbach et al., EMNLP 2018).

Reproduces the original ``data/chair/chair.py`` but in pure Python 3 (no
``pattern.en``).  Output metrics:

  * CHAIR_s : sentence-level hallucination rate
              (#captions with >=1 hallucinated obj) / (#captions)
  * CHAIR_i : instance-level hallucination rate
              (#hallucinated obj mentions) / (#total obj mentions)

We additionally report Recall and avg_len.  This matches the SAE paper which
reports CS / CI / R / P / F1 (Tab. 1).

Following the SAE protocol the prompt is "Please describe this image in
detail.", greedy decoding, max_new_tokens=512, on a fixed random subset of
COCO val2014 (size 500 by default; SAE uses 1000).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import nltk
import torch
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from PIL import Image

from ..models.llava_wrapper import LlavaWrapper
from ..utils.common import PROJECT_ROOT, load_config, set_seed
from ..decoding.baseline import greedy_decode, sample_decode
from ..decoding.vcd import vcd_decode
from ..decoding.aif import aif_decode
# (legacy ours_v3 — lazy-imported inside make_decoder when method=='ours')

CHAIR_PROMPT = "Please describe this image in detail."

# ---------------------------------------------------------------------------
# Synonym tables (mirrors the original chair.py special-casing)
# ---------------------------------------------------------------------------
COCO_DOUBLE_WORDS = [
    "motor bike", "motor cycle", "air plane", "traffic light", "street light",
    "traffic signal", "stop light", "fire hydrant", "stop sign",
    "parking meter", "suit case", "sports ball", "baseball bat",
    "baseball glove", "tennis racket", "wine glass", "hot dog", "cell phone",
    "mobile phone", "teddy bear", "hair drier", "potted plant", "bow tie",
    "laptop computer", "stove top oven", "home plate", "train track",
]
ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                "bear", "zebra", "giraffe", "animal", "cub"]
VEHICLE_WORDS = ["jet", "train"]


def _build_double_word_dict() -> Dict[str, str]:
    d = {w: w for w in COCO_DOUBLE_WORDS}
    for a in ANIMAL_WORDS:
        d[f"baby {a}"] = a
        d[f"adult {a}"] = a
    for v in VEHICLE_WORDS:
        d[f"passenger {v}"] = v
    d["bow tie"] = "tie"
    d["toilet seat"] = "toilet"
    d["wine glas"] = "wine glass"
    return d


# ---------------------------------------------------------------------------
# Singularization (pattern.en's singularize → wordnet lemmatizer)
# ---------------------------------------------------------------------------
class _Lemmatizer:
    def __init__(self):
        for pkg in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
            try:
                nltk.data.find(f"corpora/{pkg}")
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
        self.lemma = WordNetLemmatizer()

    def __call__(self, w: str) -> str:
        return self.lemma.lemmatize(w, pos="n")


# ---------------------------------------------------------------------------
# CHAIR object extractor + matcher
# ---------------------------------------------------------------------------
class CHAIR:
    def __init__(self, synonyms_file: Path):
        synonyms = [
            line.strip().split(", ")
            for line in synonyms_file.read_text().splitlines() if line.strip()
        ]
        self.mscoco_objects: Set[str] = set()
        self.inverse_synonym: Dict[str, str] = {}   # synonym -> canonical
        for syn_group in synonyms:
            for s in syn_group:
                self.mscoco_objects.add(s)
                self.inverse_synonym[s] = syn_group[0]
        self.double_word_dict = _build_double_word_dict()
        self.singularize = _Lemmatizer()

    # ------------------------------------------------------------------
    # Convert a free-form caption to the COCO objects mentioned in it
    # ------------------------------------------------------------------
    def caption_to_objects(self, caption: str) -> Tuple[List[str], List[str]]:
        words = word_tokenize(caption.lower())
        words = [self.singularize(w) for w in words]

        # collapse double words first
        i, collapsed = 0, []
        while i < len(words):
            two = " ".join(words[i:i + 2])
            if two in self.double_word_dict:
                collapsed.append(self.double_word_dict[two])
                i += 2
            else:
                collapsed.append(words[i])
                i += 1
        words = collapsed
        if "toilet" in words and "seat" in words:
            words = [w for w in words if w != "seat"]

        present = [w for w in words if w in self.mscoco_objects]
        node_words = [self.inverse_synonym[w] for w in present]
        return present, node_words

    # ------------------------------------------------------------------
    # Build per-image GT object set from MSCOCO instances + captions
    # ------------------------------------------------------------------
    def build_gt(self, instances_path: Path, captions_path: Path,
                 image_ids: Sequence[int]) -> Dict[int, Set[str]]:
        wanted = set(int(x) for x in image_ids)
        gt: Dict[int, Set[str]] = {i: set() for i in wanted}

        # 1) instances (segmentation masks)
        with open(instances_path) as f:
            inst = json.load(f)
        id_to_name = {c["id"]: c["name"] for c in inst["categories"]}
        for ann in inst["annotations"]:
            iid = ann["image_id"]
            if iid in wanted:
                cat_name = id_to_name[ann["category_id"]]
                if cat_name in self.inverse_synonym:
                    gt[iid].add(self.inverse_synonym[cat_name])

        # 2) ground truth captions
        with open(captions_path) as f:
            caps = json.load(f)
        for ann in caps["annotations"]:
            iid = ann["image_id"]
            if iid in wanted:
                _, node_words = self.caption_to_objects(ann["caption"])
                gt[iid].update(node_words)
        return gt


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------
def _coco_id_from_filename(name: str) -> int:
    m = re.search(r"COCO_val2014_(\d+)\.jpg", name)
    if not m:
        raise ValueError(f"cannot parse coco id from {name}")
    return int(m.group(1))


def sample_image_ids(image_dir: Path, n: int, seed: int) -> List[Tuple[int, str]]:
    files = sorted(p.name for p in image_dir.glob("COCO_val2014_*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(files)
    chosen = files[:n]
    return [(_coco_id_from_filename(f), f) for f in chosen]


# ---------------------------------------------------------------------------
# Decoder dispatcher (CHAIR uses greedy by default per SAE Tab. 1)
# ---------------------------------------------------------------------------
# Early prototypes whose decoder modules are not part of this release.
_UNRELEASED_METHODS = ("ours_maskk", "ours_penalty", "ours_penalty_sent",
                       "ours_combined", "ssl")

def make_decoder(method: str, wrapper: LlavaWrapper, cfg: dict, segmenter=None):
    vc = cfg["vcd"]
    ours = cfg["ours"]
    aifc = cfg["aif"]
    max_new = cfg["benchmarks"]["chair"]["max_new_tokens"]

    if method in _UNRELEASED_METHODS:
        raise ValueError(f"method {method!r} is an early prototype that is not "
                         "part of the paper and is not included in this release")

    if method == "baseline":
        return lambda img, q: greedy_decode(wrapper, img, q,
                                            max_new_tokens=max_new)
    if method == "baseline_sample":
        return lambda img, q: sample_decode(wrapper, img, q,
                                            max_new_tokens=max_new)
    if method == "ours_pmi":
        from ..decoding.ours_pmi import ours_pmi_decode
        ours = cfg["ours"]
        return lambda img, q: ours_pmi_decode(
            wrapper, img, q, max_new_tokens=max_new,
            alpha=ours.get("pmi_alpha", 1.0), beta=ours.get("beta", 0.1),
            sampling="greedy")
    if method == "vcd":
        # SAE Tab. 1 reports VCD with sampling; for free-form CHAIR
        # we follow the VCD official setup (direct sampling).
        return lambda img, q: vcd_decode(wrapper, img, q,
                                         max_new_tokens=max_new,
                                         alpha=vc["alpha"], beta=vc["beta"],
                                         noise_step=vc["noise_steps_pope"],
                                         sampling="direct")
    if method == "aif":
        return lambda img, q: aif_decode(wrapper, img, q,
                                         max_new_tokens=max_new,
                                         sampling="greedy",
                                         mask_ratios=tuple(aifc["mask_ratios"]),
                                         max_ratio=aifc.get("max_ratio", 0.5))
    if method == "ours":
        from ..decoding.ours_v3 import ours_v3_decode
        if segmenter is None:
            raise ValueError("ours requires PanopticSegmenter")
        return lambda img, q: ours_v3_decode(wrapper, segmenter, img, q,
                                             max_new_tokens=max_new,
                                             alpha=ours["alpha"],
                                             beta=ours["beta"],
                                             lookahead=ours["lookahead"],
                                             sampling="greedy")
    if method == "ours_v4":
        from ..decoding.ours_v4 import ours_v4_decode
        if segmenter is None:
            raise ValueError("ours_v4 requires PanopticSegmenter")
        boost = ours.get("v4_boost_factor", 2.0)
        return lambda img, q: ours_v4_decode(wrapper, segmenter, img, q,
                                             max_new_tokens=max_new,
                                             boost_factor=boost,
                                             lookahead=ours["lookahead"])
    if method == "ours_msb":
        from ..decoding.ours_msb import ours_msb_decode
        if segmenter is None:
            raise ValueError("ours_msb requires PanopticSegmenter")
        boost = ours.get("msb_boost_factor", 1.8)
        topk = ours.get("msb_top_k", 2)
        return lambda img, q: ours_msb_decode(wrapper, segmenter, img, q,
                                              max_new_tokens=max_new,
                                              boost_factor=boost, top_k=topk,
                                              lookahead=ours["lookahead"])
    if method == "ours_msb_rolling":
        from ..decoding.ours_msb_rolling import ours_msb_rolling_decode
        if segmenter is None:
            raise ValueError("ours_msb_rolling requires PanopticSegmenter")
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
            raise ValueError("ours_penalty requires PanopticSegmenter")
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
            raise ValueError("ours_msb_sent requires PanopticSegmenter")
        boost = ours.get("msb_boost_factor", 1.8)
        topk = ours.get("msb_top_k", 2)
        return lambda img, q: ours_msb_decode(
            wrapper, segmenter, img, q,
            max_new_tokens=max_new,
            boost_factor=boost, top_k=topk,
            lookahead=ours["lookahead"],
            use_sentence_lookahead=True)
    if method in ("ours_sbc", "ours_sbc_v2", "ours_pmi_guard", "ours_lazy", "ours_lazy_attn"):
        from ..decoding.ours_sbc import ours_sbc_decode
        if segmenter is None:
            raise ValueError(f"{method} requires PanopticSegmenter")
        if method == "ours_sbc_v2": gv = "v2"
        elif method == "ours_pmi_guard": gv = "pmi_guard_only"
        elif method == "ours_lazy": gv = "lazy"
        elif method == "ours_lazy_attn": gv = "lazy_attn"
        else: gv = "v3"
        return lambda img, q: ours_sbc_decode(
            wrapper, segmenter, img, q,
            max_new_tokens=max_new,
            boost_factor=ours.get("msb_boost_factor", 1.8),
            top_k=ours.get("msb_top_k", 2),
            use_sentence_lookahead=True,
            pmi_alpha=ours.get("pmi_alpha", 1.0), beta=ours.get("beta", 0.1),
            gate_version=gv,
            tau_mid=ours.get("sbc_tau_mid", 0.5),
            tau_lo=ours.get("sbc_tau_lo", 0.25),
            tau_hi=ours.get("sbc_tau_hi", 0.75),
            image_margin_thresh=ours.get("image_margin_thresh", 0.5),
            max_segments=ours.get("max_segments", 6),
            return_route=True)
    if method == "ours_penalty_sent":
        from ..decoding.ours_penalty import ours_penalty_decode
        if segmenter is None:
            raise ValueError("ours_penalty_sent requires PanopticSegmenter")
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
            raise ValueError("ours_combined requires PanopticSegmenter")
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
# Main eval
# ---------------------------------------------------------------------------
@dataclass
class CHAIREvalResult:
    method: str
    n_images: int
    chair_s: float
    chair_i: float
    recall: float
    avg_len: float
    seconds: float


def run(method: str, wrapper: LlavaWrapper, cfg: dict, segmenter=None,
        out_dir: Path | None = None) -> CHAIREvalResult:
    cc = cfg["benchmarks"]["chair"]
    image_dir = PROJECT_ROOT / cc["image_dir"]
    ann_dir = PROJECT_ROOT / cc["annotations_dir"]
    syn_path = PROJECT_ROOT / cc["synonyms_file"]
    n_images = cc["n_images"]
    seed = cc["seed"]

    print(f"loading CHAIR ({n_images} images)...")
    chair = CHAIR(syn_path)
    sampled = sample_image_ids(image_dir, n_images, seed)
    image_ids = [iid for iid, _ in sampled]
    print(f"building ground-truth from {ann_dir}...")
    gt = chair.build_gt(ann_dir / "instances_val2014.json",
                        ann_dir / "captions_val2014.json", image_ids)

    decoder = make_decoder(method, wrapper, cfg, segmenter=segmenter)
    set_seed(1234)

    captions = []
    n_caps, n_hallu_caps, n_hallu_words, n_total_words = 0, 0, 0, 0
    n_recall_hits, n_recall_total = 0, 0
    total_len = 0
    t0 = time.time()
    for i, (iid, fname) in enumerate(sampled):
        img = Image.open(image_dir / fname).convert("RGB")
        result = decoder(img, CHAIR_PROMPT)
        if isinstance(result, tuple):
            text, route = result
        else:
            text, route = result, None
        present, node_words = chair.caption_to_objects(text)
        gt_set = gt[iid]
        hallu = [w for w in node_words if w not in gt_set]
        n_caps += 1
        if hallu:
            n_hallu_caps += 1
        n_hallu_words += len(hallu)
        n_total_words += len(node_words)
        if gt_set:
            n_recall_hits += len(set(node_words) & gt_set)
            n_recall_total += len(gt_set)
        total_len += len(text.split())
        captions.append({"image_id": iid, "image": fname, "caption": text,
                         "gt_objects": sorted(gt_set),
                         "generated_objects": node_words,
                         "hallucinated_objects": hallu,
                         "route": route})
        if (i + 1) % 50 == 0:
            print(f"    [{method}] {i+1}/{len(sampled)}  "
                  f"CS={n_hallu_caps/n_caps*100:.1f}  "
                  f"CI={(n_hallu_words/max(1,n_total_words))*100:.1f}")
        if i % 10 == 0:
            gc.collect(); torch.cuda.empty_cache()

    chair_s = n_hallu_caps / max(1, n_caps)
    chair_i = n_hallu_words / max(1, n_total_words)
    recall = n_recall_hits / max(1, n_recall_total)
    avg_len = total_len / max(1, n_caps)
    res = CHAIREvalResult(method=method, n_images=len(sampled),
                          chair_s=chair_s, chair_i=chair_i, recall=recall,
                          avg_len=avg_len, seconds=time.time() - t0)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"raw_{method}.jsonl").write_text(
            "\n".join(json.dumps(c) for c in captions))
        (out_dir / f"summary_{method}.json").write_text(json.dumps({
            "method": method, "n_images": res.n_images,
            "chair_s": chair_s, "chair_i": chair_i, "recall": recall,
            "avg_len": avg_len, "seconds": res.seconds,
        }, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["baseline", "vcd", "aif", "opera", "pai", "ssl",
                             "ours_pmi", "ours_msb_sent", "ours_sbc",
                             "ours_sbc_v2", "ours_pmi_guard", "ours_lazy", "ours_lazy_attn"])
    ap.add_argument("--n-images", type=int, default=None,
                    help="Override the number of images.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default="results/chair")
    ap.add_argument("--lookahead", type=int, default=None,
                    help="Override the legacy fixed lookahead L (legacy fixed-L methods only; SBC and MSB-sent use a sentence lookahead of <=32 tokens)")
    ap.add_argument("--model", default=None,
                    help="Override model id (e.g. Qwen/Qwen2-VL-7B-Instruct).")
    ap.add_argument("--load-8bit", action="store_true",
                    help="Load the model in 8-bit (for 13B on a 24GB GPU).")
    ap.add_argument("--device-map", default=None,
                    help="HF device_map for fp16 multi-GPU split (e.g., 'auto', 'balanced'). Use for 13B fp16 across 2× 24GB.")
    ap.add_argument("--boost-factor", type=float, default=None,
                    help="Override SBC/MSB boost_factor (paper default 1.8, from configs/default.yaml).")
    ap.add_argument("--image-margin-thresh", type=float, default=None,
                    help="SBC image-margin guard: skip PMI when image-conditioned top-token margin exceeds this (0=off).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model is not None:
        cfg["model"]["name"] = args.model
        print(f"[override] model = {args.model}")
    if args.n_images is not None:
        cfg["benchmarks"]["chair"]["n_images"] = args.n_images
    if args.lookahead is not None:
        cfg["ours"]["lookahead"] = args.lookahead
    if args.boost_factor is not None:
        cfg.setdefault("ours", {})["msb_boost_factor"] = args.boost_factor
        print(f"[override] msb_boost_factor = {args.boost_factor}")
    if args.image_margin_thresh is not None:
        cfg.setdefault("ours", {})["image_margin_thresh"] = args.image_margin_thresh
        print(f"[override] image_margin_thresh = {args.image_margin_thresh}")
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
    if args.method in ("ours_sbc", "ours_sbc_v2", "ours_msb_sent", "ours_pmi_guard", "ours_lazy", "ours_lazy_attn"):  # paper: SBC + MSB-only
        from ..utils.segmentation import PanopticSegmenter
        print("loading Mask2Former...")
        segmenter = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                                      dtype=getattr(torch, cfg["model"]["dtype"]))

    res = run(args.method, wrapper, cfg, segmenter=segmenter, out_dir=out_dir)
    print(f"\n=== {args.method} ===")
    print(f"  N            : {res.n_images}")
    print(f"  CHAIR_s (CS) : {res.chair_s*100:5.2f}")
    print(f"  CHAIR_i (CI) : {res.chair_i*100:5.2f}")
    print(f"  Recall       : {res.recall*100:5.2f}")
    print(f"  Avg len      : {res.avg_len:5.1f}")
    print(f"  Time         : {res.seconds:5.1f}s")


if __name__ == "__main__":
    main()
