"""App K (review): wall-clock decode-time overhead per method.

For each method, decode N POPE random questions on LLaVA-1.5-7B and report:
  - mean seconds per question
  - mean seconds per generated token
  - mean tokens per question
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmarks.pope import make_decoder, POPE_QUESTION_SUFFIX
from src.models.llava_wrapper import LlavaWrapper
from src.utils.common import load_config


def _ntokens(wrapper, text):
    return len(wrapper.tokenizer.encode(text, add_special_tokens=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--methods", nargs="+",
                    default=["baseline_greedy", "vcd_greedy", "aif",
                             "opera", "pai", "ours_sbc"])
    ap.add_argument("--out", default="results/diagnostics_latency.csv")
    args = ap.parse_args()

    cfg = load_config(None)
    wrapper = LlavaWrapper(model_name="llava-hf/llava-1.5-7b-hf",
                           dtype=torch.float16, attn_implementation="eager")
    cfg["ours"]["msb_boost_factor"] = 1.8
    cfg["ours"]["image_margin_thresh"] = 0.5
    cfg["ours"]["sbc_tau_mid"] = 0.5

    seg = None
    if any(m.startswith("ours_") for m in args.methods):
        from src.utils.segmentation import PanopticSegmenter
        seg = PanopticSegmenter(
            model_name=cfg["mask2former"]["name"], dtype=torch.float16)

    image_dir = Path(__file__).resolve().parents[1] / "data/coco/val2014"
    with open(Path(__file__).resolve().parents[1] / "data/POPE/coco_pope_random.json") as f:
        questions = [json.loads(l) for l in f if l.strip()][:args.n]

    rows = []
    for m in args.methods:
        dec = make_decoder(m, wrapper, cfg, segmenter=seg)
        t0 = time.perf_counter()
        n_tok = 0
        n_done = 0
        for q in questions:
            img = Image.open(image_dir / q["image"]).convert("RGB")
            out = dec(img, q["text"] + POPE_QUESTION_SUFFIX)
            if isinstance(out, tuple):
                out = out[0]
            n_tok += _ntokens(wrapper, out)
            n_done += 1
        sec = time.perf_counter() - t0
        sec_per_q = sec / max(1, n_done)
        sec_per_tok = sec / max(1, n_tok)
        toks_per_q = n_tok / max(1, n_done)
        print(f"{m:18s}  N={n_done}  sec/q={sec_per_q:6.3f}  "
              f"sec/tok={sec_per_tok:6.3f}  tok/q={toks_per_q:5.2f}")
        rows.append((m, n_done, sec_per_q, sec_per_tok, toks_per_q))
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("method,n,sec_per_question,sec_per_token,tokens_per_question\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


if __name__ == "__main__":
    main()
