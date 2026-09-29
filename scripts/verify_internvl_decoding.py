"""Pre-flight verification for the InternVL3-8B decoding run (camera-ready
third-family evaluation). GPU 1 only.

  (a) wrapper greedy_decode == HF model.generate(do_sample=False) on 5 prompts
  (b) MSB: a 4-D mask with b=1 reproduces the 2-D path exactly; b=1.8
      raises attention mass on the boosted visual columns (text-query rows,
      every layer) and changes the logits; step masks work in decode_step
  (g) grid order: occluding one image quadrant changes the projected visual
      tokens of that quadrant most (checks row-major 16x16 mapping)
  (c) blank-image / PMI path runs (ours_pmi_decode + SBC PMI route)
  (d) SBC (main-table config) route distribution on N POPE + M CHAIR items,
      plus per-item latency for baseline and SBC (throughput for the ETA)

  CUDA_VISIBLE_DEVICES=1 python scripts/verify_internvl_decoding.py \
      --out logs/internvl_verify.json
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.common import load_config, PROJECT_ROOT          # noqa: E402
from src.models import build_wrapper                            # noqa: E402
from src.decoding.baseline import greedy_decode                 # noqa: E402
from src.decoding.ours_v4 import (_build_boost_mask,             # noqa: E402
                                  _build_step_boost_mask,
                                  _segment_to_visual_token_indices)
from src.decoding.ours_pmi import ours_pmi_decode               # noqa: E402
from src.benchmarks.pope import POPE_QUESTION_SUFFIX, make_decoder  # noqa: E402
from src.benchmarks.chair import CHAIR_PROMPT, sample_image_ids  # noqa: E402
from src.benchmarks import chair as chair_mod                   # noqa: E402

MODEL = "OpenGVLab/InternVL3-8B-hf"


def pope_items(n, setting="random"):
    rows = [json.loads(l) for l in open(PROJECT_ROOT / "data/POPE" /
                                         f"coco_pope_{setting}.json") if l.strip()]
    return rows[:n]


def load_img(name):
    return Image.open(PROJECT_ROOT / "data/coco/val2014" / name).convert("RGB")


@torch.no_grad()
def check_greedy(w, ev):
    items = pope_items(3)
    qs = [(load_img(r["image"]), r["text"] + POPE_QUESTION_SUFFIX, 64)
          for r in items]
    sampled = sample_image_ids(PROJECT_ROOT / "data/coco/val2014", 1000, 42)
    for _, fn in sampled[:2]:
        qs.append((load_img(fn), CHAIR_PROMPT, 128))
    res = []
    for img, q, mx in qs:
        ours = greedy_decode(w, img, q, max_new_tokens=mx)
        enc = w.prepare_inputs(img, q)
        g = w.model.generate(**enc, do_sample=False, max_new_tokens=mx,
                             num_beams=1)
        ref = w.tokenizer.decode(g[0, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
        res.append({"q": q[:60], "equal": ours == ref,
                    "ours": ours[:200], "hf": ref[:200]})
        print("  [greedy]", ours == ref, repr(ours[:80]))
    ev["a_greedy_equiv"] = {"n": len(res),
                            "n_equal": sum(r["equal"] for r in res),
                            "items": res}


@torch.no_grad()
def check_msb(w, ev):
    r = pope_items(1)[0]
    img = load_img(r["image"])
    enc = w.prepare_inputs(img, r["text"] + POPE_QUESTION_SUFFIX)
    ids, pv = enc["input_ids"], enc["pixel_values"]
    S = ids.shape[1]
    vis = w.visual_token_positions(ids)
    grid = w.visual_grid(ids)
    text_start = vis[-1] + 1
    # boosted columns: left-half segment mask -> tokens with >=30% coverage
    m = np.zeros((img.height, img.width), dtype=bool)
    m[:, : img.width // 2] = True
    boost = _segment_to_visual_token_indices(m, vis, grid=grid)
    local = [vis.index(p) for p in boost]
    exp_local = [rr * grid[1] + c for rr in range(grid[0])
                 for c in range(grid[1] // 2)]
    out2 = w.prefill(ids, pv, enc["attention_mask"], output_attentions=True)
    m1 = _build_boost_mask(S, boost, text_start, 1.0, w.device, w.dtype)
    out1 = w.prefill(ids, pv, m1, output_attentions=True)
    mb = _build_boost_mask(S, boost, text_start, 1.8, w.device, w.dtype)
    outb = w.prefill(ids, pv, mb, output_attentions=True)

    def mass(att):  # (layers) mean over heads & text rows of mass on boost cols
        per = []
        for a in att:
            a = a[0].float()                       # (H,S,S)
            rows = a[:, text_start:, :]
            per.append(float(rows[..., boost].sum(-1).mean()))
        return per
    base_m, one_m, b_m = mass(out2.attentions), mass(out1.attentions), \
        mass(outb.attentions)
    l2 = out2.logits[0, -1].float()
    l1 = out1.logits[0, -1].float()
    lb = outb.logits[0, -1].float()
    # one decode step with the per-step boost mask
    nxt = lb.argmax().view(1, 1)
    step = _build_step_boost_mask(S, boost, 1.8, w.device, w.dtype)
    ls, _ = w.decode_step(nxt, outb.past_key_values, attention_mask=step)
    step1 = _build_step_boost_mask(S, boost, 1.0, w.device, w.dtype)
    ls1, _ = w.decode_step(nxt, out1.past_key_values, attention_mask=step1)
    ls_ref, _ = w.decode_step(nxt, out2.past_key_values)
    ev["b_msb"] = {
        "seq_len": S, "n_visual": len(vis), "grid": list(grid),
        "text_start": text_start, "n_boost_cols": len(boost),
        "boost_cols_equal_left_half_rowmajor": sorted(local) == exp_local,
        "mass_2d_mean_over_layers": float(np.mean(base_m)),
        "mass_4d_b1_mean_over_layers": float(np.mean(one_m)),
        "mass_4d_b1.8_mean_over_layers": float(np.mean(b_m)),
        "mass_ratio_b1.8_over_b1_per_layer_min": float(np.min(
            np.array(b_m) / np.array(one_m))),
        "mass_ratio_b1.8_over_b1_per_layer_max": float(np.max(
            np.array(b_m) / np.array(one_m))),
        "n_layers_mass_increased": int(sum(b > o for b, o in zip(b_m, one_m))),
        "n_layers": len(b_m),
        "logits_maxabs_4d_b1_vs_2d": float((l1 - l2).abs().max()),
        "logits_maxabs_b1.8_vs_b1": float((lb - l1).abs().max()),
        "kl_b1.8_vs_b1_last": float(torch.sum(
            torch.softmax(l1, -1) * (torch.log_softmax(l1, -1) -
                                     torch.log_softmax(lb, -1)))),
        "step_logits_maxabs_b1_vs_2d": float(
            (ls1[0, -1].float() - ls_ref[0, -1].float()).abs().max()),
        "step_logits_maxabs_b1.8_vs_2d": float(
            (ls[0, -1].float() - ls_ref[0, -1].float()).abs().max()),
    }
    print("  [msb]", json.dumps(ev["b_msb"], indent=1))
    del out2, out1, outb
    torch.cuda.empty_cache()


@torch.no_grad()
def check_grid_order(w, ev):
    """Occlude the bottom-right quadrant; the visual tokens whose projected
    features change most should lie in the bottom-right 8x8 block."""
    r = pope_items(1)[0]
    img = load_img(r["image"]).resize((448, 448))
    arr = np.asarray(img).copy()
    arr[224:, 224:] = arr.reshape(-1, 3).mean(0).astype(arr.dtype)
    occ = Image.fromarray(arr)
    f0 = w.model.model.get_image_features(pixel_values=w._pixels(img),
                                          vision_feature_layer=w.model.config.vision_feature_layer,
                                          vision_feature_select_strategy=w.model.config.vision_feature_select_strategy)
    f1 = w.model.model.get_image_features(pixel_values=w._pixels(occ),
                                          vision_feature_layer=w.model.config.vision_feature_layer,
                                          vision_feature_select_strategy=w.model.config.vision_feature_select_strategy)
    f0 = getattr(f0, "pooler_output", f0)
    f1 = getattr(f1, "pooler_output", f1)
    d = (f0 - f1).float().norm(dim=-1)[0].view(w.grid_h, w.grid_w).cpu()
    h = w.grid_h // 2
    br = float(d[h:, h:].mean())
    rest = float(torch.cat([d[:h, :].flatten(), d[h:, :h].flatten()]).mean())
    top64 = torch.topk(d.flatten(), 64).indices
    in_br = int(sum(1 for i in top64.tolist()
                    if i // w.grid_w >= h and i % w.grid_w >= h))
    # transposed (column-major) hypothesis would give the same quadrant, so
    # also test an off-diagonal quadrant: top-right
    arr2 = np.asarray(img).copy()
    arr2[:224, 224:] = arr2.reshape(-1, 3).mean(0).astype(arr2.dtype)
    f2 = w.model.model.get_image_features(pixel_values=w._pixels(Image.fromarray(arr2)),
                                          vision_feature_layer=w.model.config.vision_feature_layer,
                                          vision_feature_select_strategy=w.model.config.vision_feature_select_strategy)
    f2 = getattr(f2, "pooler_output", f2)
    d2 = (f0 - f2).float().norm(dim=-1)[0].view(w.grid_h, w.grid_w).cpu()
    tr = float(d2[:h, h:].mean())
    bl = float(d2[h:, :h].mean())
    ev["g_grid_order"] = {"occl_bottom_right_mean_change_in_BR": br,
                          "occl_bottom_right_mean_change_elsewhere": rest,
                          "top64_changed_tokens_in_BR": in_br,
                          "occl_top_right_change_in_TR_block(row-major)": tr,
                          "occl_top_right_change_in_BL_block(transposed)": bl}
    print("  [grid]", ev["g_grid_order"])


@torch.no_grad()
def check_pmi(w, ev):
    out = []
    for r in pope_items(3, "adversarial"):
        img = load_img(r["image"])
        t0 = time.time()
        txt = ours_pmi_decode(w, img, r["text"] + POPE_QUESTION_SUFFIX,
                              max_new_tokens=64, alpha=1.0, beta=0.1)
        out.append({"q": r["text"], "gt": r["label"], "pmi": txt,
                    "sec": time.time() - t0})
    enc = w.prepare_inputs(img, "x")
    blank = w.prefill(enc["input_ids"], torch.zeros_like(enc["pixel_values"]),
                      enc["attention_mask"])
    ev["c_pmi"] = {"items": out,
                   "blank_logits_finite": bool(torch.isfinite(blank.logits).all())}
    print("  [pmi]", ev["c_pmi"])


def check_sbc(w, cfg, ev, n_pope, n_chair):
    from src.utils.segmentation import PanopticSegmenter
    seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                            dtype=getattr(torch, cfg["model"]["dtype"]))
    dec_sbc = make_decoder("ours_sbc", w, cfg, segmenter=seg)
    dec_base = make_decoder("baseline_greedy", w, cfg)
    pope_rows = []
    per = max(1, n_pope // 3)
    for s in ("random", "popular", "adversarial"):
        pope_rows += [(s, r) for r in pope_items(per * 20, s)[::20][:per]]
    rec = {"pope": [], "chair": []}
    for s, r in pope_rows:
        img = load_img(r["image"])
        q = r["text"] + POPE_QUESTION_SUFFIX
        t0 = time.time(); b = dec_base(img, q); tb = time.time() - t0
        t0 = time.time(); txt, route = dec_sbc(img, q); ts = time.time() - t0
        rec["pope"].append({"split": s, "gt": r["label"], "base": b,
                            "sbc": txt, "route": route, "t_base": tb,
                            "t_sbc": ts})
        print(f"  [pope/{s}] gt={r['label']} base={b!r} sbc={txt!r} "
              f"route={route} tb={tb:.2f}s ts={ts:.2f}s", flush=True)
    ccfg = dict(cfg)
    ccfg["benchmarks"] = dict(cfg["benchmarks"])
    dec_sbc_c = chair_mod.make_decoder("ours_sbc", w, cfg, segmenter=seg)
    dec_base_c = chair_mod.make_decoder("baseline", w, cfg)
    sampled = sample_image_ids(PROJECT_ROOT / "data/coco/val2014", 1000, 42)
    for _, fn in sampled[:n_chair]:
        img = load_img(fn)
        t0 = time.time(); b = dec_base_c(img, CHAIR_PROMPT); tb = time.time() - t0
        t0 = time.time(); txt, route = dec_sbc_c(img, CHAIR_PROMPT)
        ts = time.time() - t0
        rec["chair"].append({"image": fn, "base": b, "sbc": txt,
                             "route": route, "t_base": tb, "t_sbc": ts,
                             "len_base": len(b.split()),
                             "len_sbc": len(txt.split())})
        print(f"  [chair] {fn} route={route} tb={tb:.1f}s ts={ts:.1f}s "
              f"len {len(b.split())}/{len(txt.split())} "
              f"same={b == txt}", flush=True)
    summ = {}
    for k in ("pope", "chair"):
        rows = rec[k]
        summ[k] = {"n": len(rows),
                   "routes": dict(Counter(r["route"] for r in rows)),
                   "t_base_mean": float(np.mean([r["t_base"] for r in rows])),
                   "t_sbc_mean": float(np.mean([r["t_sbc"] for r in rows])),
                   "n_output_changed": int(sum(r["base"] != r["sbc"]
                                               for r in rows))}
    summ["pope"]["acc_base"] = float(np.mean(
        [("yes" in r["base"].lower()) == (r["gt"] == "yes") for r in rec["pope"]]))
    summ["pope"]["acc_sbc"] = float(np.mean(
        [("yes" in r["sbc"].lower()) == (r["gt"] == "yes") for r in rec["pope"]]))
    ev["d_sbc"] = {"summary": summ, "items": rec}
    print("  [sbc]", json.dumps(summ, indent=1))
    ev["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/internvl_verify.json")
    ap.add_argument("--n-pope", type=int, default=30)
    ap.add_argument("--n-chair", type=int, default=10)
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "GPU 1 only"
    cfg = load_config(None)
    cfg["model"]["name"] = MODEL
    w = build_wrapper(MODEL, dtype=getattr(torch, cfg["model"]["dtype"]),
                      attn_implementation=cfg["model"]["attn_implementation"])
    ev = {"model": MODEL, "dtype": str(w.dtype),
          "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    out = PROJECT_ROOT / a.out
    skip = set(a.skip.split(","))
    for name, fn in (("a", lambda: check_greedy(w, ev)),
                     ("b", lambda: check_msb(w, ev)),
                     ("g", lambda: check_grid_order(w, ev)),
                     ("c", lambda: check_pmi(w, ev)),
                     ("d", lambda: check_sbc(w, cfg, ev, a.n_pope, a.n_chair))):
        if name in skip:
            continue
        print(f"== check {name}", flush=True)
        fn()
        out.write_text(json.dumps(ev, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
