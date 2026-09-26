"""SBC v3 H distribution on POPE — for each sampled question, record:
  H (= normalized entropy of softmax(φ)),
  blank_match (only when H ≥ τ_mid),
  route ∈ {pmi, msb, fallback},
  K (#segments), and the lookahead span length.

Then bucket by current single-threshold rule:
  H < τ_mid                          → MSB  (over-concentrated)
  H ≥ τ_mid AND blank_match          → PMI  (over-spread)
  H ≥ τ_mid AND not blank_match      → MSB  (PMI prerequisite fails)
  <2 segments OR no usable span      → fallback (greedy)

Saves results/sbc_h_dist_pope.json (and per-split jsonl rows for plotting).

Usage:  CUDA_VISIBLE_DEVICES=1 python scripts/sbc_h_distribution.py [N_PER_SPLIT]
        default N_PER_SPLIT = 200  (so 600 total)
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.llava_wrapper import LlavaWrapper                # noqa: E402
from src.utils.segmentation import PanopticSegmenter             # noqa: E402
from src.utils.common import load_config                         # noqa: E402
from src.decoding.ours_sbc import (_lookahead_with_logp,         # noqa: E402
                                   _blank_span_matches,
                                   _norm_entropy)
from src.decoding.ours_msb import _shap_phis_batched              # noqa: E402
from src.benchmarks.pope import POPE_QUESTION_SUFFIX             # noqa: E402

TAU_MID = 0.5


@torch.no_grad()
def measure(wrapper, segmenter, image, question):
    """Run only the SBC v3 *measurement* path (no generation) and return the
    routing decision + intermediate stats."""
    enc = wrapper.prepare_inputs(image, question)
    iid = enc["input_ids"]; pv = enc["pixel_values"]; am = enc.get("attention_mask")
    segs = segmenter.segment(image, min_area_frac=0.01, max_segments=6)
    K = len(segs)
    if K < 2:
        return {"route": "fallback", "reason": "<2 seg", "K": K}
    span, *_ = _lookahead_with_logp(wrapper, iid, pv, am, True, 8, max_steps=32)
    if not span:
        return {"route": "fallback", "reason": "no span", "K": K}
    phis = _shap_phis_batched(wrapper, image, segs, iid, pv, am, span)
    H = _norm_entropy(phis)
    out = {"H": float(H), "K": K, "span_len": len(span),
           "phis": [float(x) for x in phis]}
    if H >= TAU_MID:
        match, *_ = _blank_span_matches(
            wrapper, iid, torch.zeros_like(pv), am, span, True, 8, max_steps=32)
        out["blank_match"] = bool(match)
        out["route"] = "pmi" if match else "msb"
    else:
        out["blank_match"] = None
        out["route"] = "msb"
    return out


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    cfg = load_config()
    pb = cfg["benchmarks"]["pope"]
    img_dir = ROOT / pb["image_dir"]
    data_dir = ROOT / pb["data_dir"]

    print(f"loading models...  τ_mid={TAU_MID}  n_per_split={n_per}", flush=True)
    w = LlavaWrapper(model_name=cfg["model"]["name"],
                     dtype=getattr(torch, cfg["model"]["dtype"]),
                     attn_implementation=cfg["model"]["attn_implementation"])
    seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                            dtype=getattr(torch, cfg["model"]["dtype"]))

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    report = {"tau_mid": TAU_MID, "splits": {}}

    rng = np.random.default_rng(0)
    for split in ("random", "popular", "adversarial"):
        qs = [json.loads(l) for l in open(data_dir / f"coco_pope_{split}.json")]
        idx = rng.choice(len(qs), size=min(n_per, len(qs)), replace=False)
        rows = []
        routes = Counter()
        t0 = time.time()
        for k, i in enumerate(idx):
            q = qs[int(i)]
            img_path = img_dir / q["image"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            r = measure(w, seg, img, q["text"] + POPE_QUESTION_SUFFIX)
            r.update(image=q["image"], question=q["text"],
                     gt=q.get("label"), qid=int(i))
            rows.append(r)
            routes[r["route"]] += 1
            if (k + 1) % 25 == 0:
                Hs = [x["H"] for x in rows if "H" in x]
                Hm = np.mean(Hs) if Hs else float("nan")
                elapsed = time.time() - t0
                eta = elapsed / (k + 1) * (len(idx) - k - 1)
                print(f"  [{split}] {k+1}/{len(idx)}  routes={dict(routes)}  "
                      f"H̄={Hm:.3f}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m",
                      flush=True)
        # also dump per-split jsonl
        with open(out_dir / f"sbc_h_dist_pope_{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        Hs = np.array([x["H"] for x in rows if "H" in x])
        report["splits"][split] = {
            "n": len(rows),
            "routes": dict(routes),
            "H_mean": float(Hs.mean()) if len(Hs) else None,
            "H_median": float(np.median(Hs)) if len(Hs) else None,
            "H_lt_tau": int((Hs < TAU_MID).sum()),
            "H_ge_tau": int((Hs >= TAU_MID).sum()),
        }
        print(f"  → {split}: n={len(rows)}  routes={dict(routes)}  "
              f"H̄={Hs.mean():.3f}  H_med={np.median(Hs):.3f}", flush=True)

    out = out_dir / "sbc_h_dist_pope.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n✓ wrote {out}")


if __name__ == "__main__":
    main()
