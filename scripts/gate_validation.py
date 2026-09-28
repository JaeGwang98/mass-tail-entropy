#!/usr/bin/env python3
"""Gate-validation analysis for SBC.

Answers the reviewer question "is the attribution-entropy gate actually doing work, or is
SBC ~= always-MSB / always-PMI?" by bracketing SBC between two reference
gates computed offline from the per-item raw logs:

  oracle gate  : route each item to whichever of {MSB, PMI} is correct
                 (upper bound on any gate)
  random gate  : route each item 50/50            (no-signal baseline)

The decisive number is SBC accuracy *on items where MSB and PMI disagree*:
if it beats 50%, the gate extracts real signal.

POPE raw: results/pope_<tag>/raw_<method>_<split>_run0.jsonl
          one json/line {qid, gt, pred, ...}
"""
from __future__ import annotations
import glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ["random", "popular", "adversarial"]


def load_split(tag: str, split: str) -> dict:
    """Return {qid: correct(bool)} for one POPE split, or {} if absent."""
    hits = glob.glob(os.path.join(ROOT, "results", tag,
                                  f"raw_*_{split}_run0.jsonl"))
    if not hits:
        return {}
    out = {}
    for line in open(hits[0]):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        gt = str(r["gt"]).strip().lower()
        pred = str(r["pred"]).strip().lower()
        out[r["qid"]] = (gt == pred)
    return out


def analyse(msb_tag: str, pmi_tag: str, sbc_tag: str, label: str):
    print(f"\n{'='*66}\n{label}\n{'='*66}")
    agg = {"n": 0, "msb": 0, "pmi": 0, "sbc": 0,
           "oracle": 0, "worst": 0, "disagree": 0, "sbc_on_disagree": 0}
    for split in SPLITS:
        m = load_split(msb_tag, split)
        p = load_split(pmi_tag, split)
        s = load_split(sbc_tag, split)
        qids = sorted(set(m) & set(p) & set(s))
        if not qids:
            print(f"  {split:12s}  (raw 누락 — skip)")
            continue
        n = len(qids)
        msb = sum(m[q] for q in qids)
        pmi = sum(p[q] for q in qids)
        sbc = sum(s[q] for q in qids)
        oracle = sum(m[q] or p[q] for q in qids)
        worst = sum(m[q] and p[q] for q in qids)
        dis = [q for q in qids if m[q] != p[q]]
        sbc_dis = sum(s[q] for q in dis)
        rnd = 0.5 * msb + 0.5 * pmi
        print(f"  {split:12s} n={n}")
        print(f"    MSB={msb/n*100:.2f}  PMI={pmi/n*100:.2f}  "
              f"SBC={sbc/n*100:.2f}")
        print(f"    random-gate={rnd/n*100:.2f}  oracle={oracle/n*100:.2f}  "
              f"(worst={worst/n*100:.2f})")
        if dis:
            print(f"    disagreement={len(dis)/n*100:.1f}%  "
                  f"SBC on disagreements={sbc_dis/len(dis)*100:.2f}%  "
                  f"(>50% => gate has signal)")
        for k, v in (("n", n), ("msb", msb), ("pmi", pmi), ("sbc", sbc),
                     ("oracle", oracle), ("worst", worst),
                     ("disagree", len(dis)), ("sbc_on_disagree", sbc_dis)):
            agg[k] += v
    n = agg["n"]
    if not n:
        print("  (집계 불가)")
        return
    msb, pmi, sbc = agg["msb"]/n, agg["pmi"]/n, agg["sbc"]/n
    orc, wst = agg["oracle"]/n, agg["worst"]/n
    rnd = 0.5*msb + 0.5*pmi
    print(f"  {'-'*60}")
    print(f"  OVERALL (n={n})")
    print(f"    MSB-only      {msb*100:.2f}")
    print(f"    PMI-only      {pmi*100:.2f}")
    print(f"    random gate   {rnd*100:.2f}   <- no-signal baseline")
    print(f"    SBC (actual)  {sbc*100:.2f}")
    print(f"    oracle gate   {orc*100:.2f}   <- upper bound")
    if orc > rnd:
        eff = (sbc - rnd) / (orc - rnd) * 100
        print(f"    gate efficiency = (SBC-random)/(oracle-random) "
              f"= {eff:.1f}%")
    if agg["disagree"]:
        sod = agg["sbc_on_disagree"] / agg["disagree"]
        print(f"    disagreement items: {agg['disagree']} "
              f"({agg['disagree']/n*100:.1f}%)")
        print(f"    SBC accuracy on disagreements = {sod*100:.2f}%  "
              f"({'signal' if sod > 0.5 else 'NO signal'})")


if __name__ == "__main__":
    # bf=1.5-era LLaVA-1.5-7B set (consistent triple).
    analyse("pope_msb_only", "pope_pmi_only", "pope_sbc_rebaseline",
            "POPE / LLaVA-1.5-7B  (bf=1.5-era; refresh on bf=1.8)")
    # Qwen2.5-VL set if a consistent triple exists.
    if all(glob.glob(os.path.join(ROOT, "results", t, "raw_*random*"))
           for t in ("pope_msb_only_qwen2_5vl", "pope_pmi_only_qwen2_5vl",
                     "pope_sbc_qwen2_5vl")):
        analyse("pope_msb_only_qwen2_5vl", "pope_pmi_only_qwen2_5vl",
                "pope_sbc_qwen2_5vl", "POPE / Qwen2.5-VL-7B  (bf=1.5-era)")
