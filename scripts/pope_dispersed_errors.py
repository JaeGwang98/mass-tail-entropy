"""Fast focused test: on the POPE questions baseline_greedy gets WRONG
(over-dispersed regime — for 1-token POPE answers ~all questions are
over-dispersed), compare how many each method *recovers*.

Baseline-error qids + baseline/PMI preds are read from the existing 500q
pilot dump (results/pope_pmi_pilot/). PMI is deterministic so its preds are
reused; only ours_maskk is re-run (it needs the segmenter).

Usage:  CUDA_VISIBLE_DEVICES=0 python scripts/pope_dispersed_errors.py
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
from src.utils.segmentation import PanopticSegmenter             # noqa: E402
from src.utils.common import load_config                         # noqa: E402
try:                                                             # noqa: E402
    from src.decoding.ours_maskk import ours_maskk_decode
except ImportError as e:  # early prototype, not part of this release
    raise SystemExit("pope_dispersed_errors.py needs the ours_maskk prototype, "
                     "which is not included in this release") from e
from src.decoding.ours_pmi import ours_pmi_decode               # noqa: E402
from src.benchmarks.pope import POPE_QUESTION_SUFFIX, parse_yes_no  # noqa: E402

PILOT = ROOT / "results" / "pope_pmi_pilot"
IMG_DIR = ROOT / "data" / "coco" / "val2014"
SPLITS = ["random", "popular", "adversarial"]
OUT = ROOT / "results" / "pope_dispersed_errors.json"


def load_split(split: str):
    base = {json.loads(l)["qid"]: json.loads(l)
            for l in open(PILOT / f"raw_baseline_greedy_{split}_run0.jsonl")}
    pmi = {json.loads(l)["qid"]: json.loads(l)
           for l in open(PILOT / f"raw_ours_pmi_{split}_run0.jsonl")}
    err_qids = [q for q, r in base.items() if r["pred"] != r["gt"].lower()]
    return base, pmi, err_qids


def main():
    cfg = load_config()
    print("loading LLaVA-1.5-7B + Mask2Former ...")
    w = LlavaWrapper(model_name=cfg["model"]["name"],
                     dtype=getattr(torch, cfg["model"]["dtype"]),
                     attn_implementation=cfg["model"]["attn_implementation"])
    seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                            dtype=getattr(torch, cfg["model"]["dtype"]))

    img_cache: dict[str, Image.Image] = {}
    report = {}
    t0 = time.time()
    for split in SPLITS:
        base, pmi, err_qids = load_split(split)
        rows = []
        for i, qid in enumerate(err_qids):
            r = base[qid]
            gt = r["gt"].lower()
            img = img_cache.get(r["image"])
            if img is None:
                img = Image.open(IMG_DIR / r["image"]).convert("RGB")
                img_cache[r["image"]] = img
            q = r["question"] + POPE_QUESTION_SUFFIX
            mk = parse_yes_no(ours_maskk_decode(w, seg, img, q,
                                                max_new_tokens=cfg["benchmarks"]["pope"]["max_new_tokens"],
                                                alpha=cfg["ours"].get("pmi_alpha", 1.0),
                                                beta=cfg["ours"].get("beta", 0.1),
                                                top_k=cfg["ours"].get("maskk_top_k", 2)))
            pm = pmi[qid]["pred"]
            rows.append(dict(qid=qid, image=r["image"], question=r["question"],
                             gt=gt, baseline=r["pred"], pmi=pm, maskk=mk))
            if (i + 1) % 25 == 0:
                print(f"  [{split}] {i+1}/{len(err_qids)}")
        n = len(rows)
        pmi_fix = sum(1 for x in rows if x["pmi"] == x["gt"])
        mk_fix = sum(1 for x in rows if x["maskk"] == x["gt"])
        both = sum(1 for x in rows if x["pmi"] == x["gt"] and x["maskk"] == x["gt"])
        only_pmi = sum(1 for x in rows if x["pmi"] == x["gt"] and x["maskk"] != x["gt"])
        only_mk = sum(1 for x in rows if x["maskk"] == x["gt"] and x["pmi"] != x["gt"])
        neither = sum(1 for x in rows if x["pmi"] != x["gt"] and x["maskk"] != x["gt"])
        agree = sum(1 for x in rows if x["pmi"] == x["maskk"])
        report[split] = dict(n_baseline_errors=n,
                             pmi_recovered=pmi_fix, maskk_recovered=mk_fix,
                             both=both, only_pmi=only_pmi, only_maskk=only_mk,
                             neither=neither, pmi_maskk_agree=agree, rows=rows)
        print(f"\n[{split}] baseline errors={n}  PMI recovers={pmi_fix}  "
              f"MaskK recovers={mk_fix}  | both={both} onlyPMI={only_pmi} "
              f"onlyMaskK={only_mk} neither={neither}  PMI==MaskK on {agree}/{n}")

    # overall
    allrows = [x for s in SPLITS for x in report[s]["rows"]]
    N = len(allrows)
    pf = sum(1 for x in allrows if x["pmi"] == x["gt"])
    mf = sum(1 for x in allrows if x["maskk"] == x["gt"])
    ag = sum(1 for x in allrows if x["pmi"] == x["maskk"])
    report["overall"] = dict(n=N, pmi_recovered=pf, maskk_recovered=mf,
                             pmi_maskk_agree=ag)
    print(f"\n=== OVERALL: {N} baseline errors | PMI recovers {pf} ({100*pf/N:.1f}%)  "
          f"MaskK recovers {mf} ({100*mf/N:.1f}%)  | PMI==MaskK on {ag}/{N} ({100*ag/N:.1f}%)  "
          f"| {time.time()-t0:.0f}s")
    OUT.write_text(json.dumps(report, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
