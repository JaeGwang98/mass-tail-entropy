"""E1 smoke test: the `lazy` gate must reproduce `no_h` (same outputs, same
routes) while skipping segmentation/occlusion on PMI-routed questions.

    python scripts/smoke_lazy_gate.py --n 40 --split random
"""
import argparse, json, sys, time
from pathlib import Path
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmarks.pope import make_decoder, POPE_QUESTION_SUFFIX
from src.models.llava_wrapper import LlavaWrapper
from src.utils.common import load_config
from src.utils.segmentation import PanopticSegmenter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--split", default="random")
    ap.add_argument("--out", default="results/smoke_lazy_gate.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(None)
    cfg["ours"]["msb_boost_factor"] = 1.8
    cfg["ours"]["image_margin_thresh"] = 0.5
    cfg["ours"]["sbc_tau_mid"] = 0.5
    wrapper = LlavaWrapper(model_name="llava-hf/llava-1.5-7b-hf",
                           dtype=torch.float16, attn_implementation="eager")
    seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"], dtype=torch.float16)

    with open(root / f"data/POPE/coco_pope_{args.split}.json") as f:
        qs = [json.loads(l) for l in f if l.strip()][:args.n]
    image_dir = root / "data/coco/val2014"

    # count segmenter calls to prove laziness
    n_seg = {"k": 0}
    orig = seg.segment
    def counted(*a, **kw):
        n_seg["k"] += 1
        return orig(*a, **kw)
    seg.segment = counted

    res = {}
    for m in ["ours_no_h", "ours_lazy"]:
        dec = make_decoder(m, wrapper, cfg, segmenter=seg)
        n_seg["k"] = 0
        torch.cuda.synchronize(); t0 = time.perf_counter()
        outs = []
        for q in qs:
            img = Image.open(image_dir / q["image"]).convert("RGB")
            o = dec(img, q["text"] + POPE_QUESTION_SUFFIX)
            out, route = (o if isinstance(o, tuple) else (o, None))
            outs.append({"out": out, "route": route})
        torch.cuda.synchronize(); sec = time.perf_counter() - t0
        res[m] = {"sec_per_q": sec / len(qs), "segment_calls": n_seg["k"], "outs": outs}
        print(f"{m:10s} sec/q={sec/len(qs):.3f}  segment_calls={n_seg['k']}/{len(qs)}", flush=True)
        torch.cuda.empty_cache()

    a, b = res["ours_no_h"]["outs"], res["ours_lazy"]["outs"]
    same_out = sum(x["out"] == y["out"] for x, y in zip(a, b))
    same_route = sum(x["route"] == y["route"] for x, y in zip(a, b))
    from collections import Counter
    print("routes no_h :", Counter(x["route"] for x in a))
    print("routes lazy :", Counter(x["route"] for x in b))
    print(f"identical outputs: {same_out}/{len(qs)}   identical routes: {same_route}/{len(qs)}")
    print(f"speedup lazy vs no_h: {res['ours_no_h']['sec_per_q']/res['ours_lazy']['sec_per_q']:.2f}x")
    res["identical_outputs"] = same_out; res["identical_routes"] = same_route
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
