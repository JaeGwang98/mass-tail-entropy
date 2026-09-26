"""Rebuttal (R3-W3): generation-quality metrics for the CHAIR captions of
every method in paper Table 1, computed from the saved raw_*.jsonl outputs.

Metrics vs the 5 COCO val2014 reference captions per image:
  BLEU-1/4, CIDEr  (pycocoevalcap, simple whitespace+punct tokenizer --
  applied identically to all methods, so relative comparison is unaffected)
plus reference-free fluency/diversity proxies:
  avg words, distinct-1, distinct-2, 4-gram in-caption repetition rate.

CPU-only. Usage:  python scripts/caption_quality.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

RUNS = {
    "llava7b": {
        "Baseline": RES / "chair_full/raw_baseline.jsonl",
        "VCD": RES / "chair_full/raw_vcd.jsonl",
        "OPERA": RES / "chair_opera_llava7b_1k/raw_opera.jsonl",
        "AIF": RES / "chair_full/raw_aif.jsonl",
        "MSB-only": RES / "chair_msb_only_bf1_8_llava7b/raw_ours_msb_sent.jsonl",
        "PMI-only": RES / "chair_pmi_only/raw_ours_pmi.jsonl",
        "SBC": RES / "chair_sbc_v3_b18_margin03_llava7b/raw_ours_sbc.jsonl",
    },
    "qwen2_5vl": {
        "Baseline": RES / "chair_baseline_greedy_qwen2_5vl/raw_baseline.jsonl",
        "VCD": RES / "chair_vcd_qwen2_5vl/raw_vcd.jsonl",
        "OPERA": RES / "chair_opera_qwen2_5vl/raw_opera.jsonl",
        "AIF": RES / "chair_aif_qwen2_5vl/raw_aif.jsonl",
        "MSB-only": RES / "chair_msb_only_bf1_8_qwen2_5vl/raw_ours_msb_sent.jsonl",
        "PMI-only": RES / "chair_pmi_only_qwen2_5vl/raw_ours_pmi.jsonl",
        "SBC": RES / "chair_sbc_v3_b18_margin03_qwen2_5vl/raw_ours_sbc.jsonl",
    },
}

_tok_re = re.compile(r"[a-z0-9]+")


def tok(s: str) -> str:
    return " ".join(_tok_re.findall(s.lower()))


def load_refs():
    ann = json.loads(
        (ROOT / "data/coco/annotations/captions_val2014.json").read_text())
    refs = {}
    for a in ann["annotations"]:
        refs.setdefault(int(a["image_id"]), []).append(tok(a["caption"]))
    return refs


def ngram_stats(caps):
    d1n = d1d = d2n = d2d = rep4 = rep4d = 0
    words_total = 0
    for c in caps:
        w = c.split()
        words_total += len(w)
        d1n += len(set(w)); d1d += max(1, len(w))
        bi = list(zip(w, w[1:]))
        d2n += len(set(bi)); d2d += max(1, len(bi))
        g4 = list(zip(w, w[1:], w[2:], w[3:]))
        if g4:
            c4 = Counter(g4)
            rep4 += sum(v - 1 for v in c4.values())
            rep4d += len(g4)
    n = max(1, len(caps))
    return {"avg_words": words_total / n,
            "distinct1": d1n / max(1, d1d),
            "distinct2": d2n / max(1, d2d),
            "rep4": rep4 / max(1, rep4d)}


def main():
    refs = load_refs()
    out_rows = []
    for model, runs in RUNS.items():
        for name, path in runs.items():
            if not path.exists():
                print(f"!! missing {path}")
                continue
            recs = [json.loads(l) for l in open(path) if l.strip()]
            gts, cands = {}, {}
            caps = []
            for r in recs:
                iid = int(r["image_id"])
                if iid not in refs:
                    continue
                c = tok(r["caption"])
                gts[iid] = refs[iid]
                cands[iid] = [c]
                caps.append(c)
            bleu, _ = Bleu(4).compute_score(gts, cands)
            cider, _ = Cider().compute_score(gts, cands)
            ns = ngram_stats(caps)
            row = {"model": model, "method": name, "n": len(cands),
                   "bleu1": bleu[0], "bleu4": bleu[3], "cider": cider, **ns}
            out_rows.append(row)
            print(f"{model:10s} {name:9s} n={len(cands):4d} "
                  f"B1={bleu[0]:.3f} B4={bleu[3]:.3f} CIDEr={cider:.3f} "
                  f"len={ns['avg_words']:5.1f} d1={ns['distinct1']:.3f} "
                  f"d2={ns['distinct2']:.3f} rep4={ns['rep4']:.4f}",
                  flush=True)
    out = RES / "rebuttal_caption_quality.json"
    out.write_text(json.dumps(out_rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
