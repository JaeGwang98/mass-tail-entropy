"""App I.2: SHAP-H vs attention-H correlation on LLaVA POPE random.

For each sample:
  - segment image with Mask2Former
  - SHAP-H = normalized entropy of softmax(LOO-occlusion phi over segments)
  - Attn-H = normalized entropy of softmax(aggregated last-token attn to visual
            positions per segment)
Write a CSV (qid, n_segments, shap_H, attn_H, baseline_pred) for later
correlation analysis.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.llava_wrapper import LlavaWrapper
from src.utils.segmentation import PanopticSegmenter
from src.decoding.ours_msb import _shap_phis_batched as shap_phis
from src.decoding.ours_v4 import _segment_to_visual_token_indices
from src.decoding.ours_sbc import _norm_entropy


def _attn_H(wrapper, segments, prompt_ids, pixel_values, attn_mask) -> float:
    """Entropy of softmax(per-segment summed last-token attention)."""
    out = wrapper.prefill(prompt_ids, pixel_values, attn_mask,
                          output_attentions=True)
    # last layer, mean over heads, last query position
    last = out.attentions[-1][0].mean(dim=0)[-1].float()  # (seq,)
    vis_pos = wrapper.visual_token_positions(prompt_ids)
    if not vis_pos:
        return float("nan")
    seg_attn = []
    grid = wrapper.visual_grid()
    for seg in segments:
        abs_idx = _segment_to_visual_token_indices(seg.mask, vis_pos, grid=grid)
        if not abs_idx:
            seg_attn.append(0.0); continue
        seg_attn.append(float(last[abs_idx].sum().item()))
    a = np.asarray(seg_attn, dtype=np.float64)
    if a.sum() <= 0:
        return float("nan")
    a = a / a.sum()
    a = np.clip(a, 1e-12, 1.0)
    return float(-(a * np.log(a)).sum() / math.log(len(a))) if len(a) > 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--split", default="random")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="results/diagnostics_attn_vs_shap_H.csv")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wrapper = LlavaWrapper(model_name=args.model, dtype=torch.float16,
                           attn_implementation="eager")
    segmenter = PanopticSegmenter(
        model_name="facebook/mask2former-swin-large-coco-panoptic",
        dtype=torch.float16)

    pope_path = Path("data/POPE") / f"coco_pope_{args.split}.json"
    with open(pope_path) as f:
        questions = [json.loads(l) for l in f if l.strip()][:args.n]

    image_dir = Path(__file__).resolve().parents[1] / "data/coco/val2014"
    rows = []
    yes_id = wrapper.tokenizer.encode(" Yes", add_special_tokens=False)[-1]
    no_id = wrapper.tokenizer.encode(" No", add_special_tokens=False)[-1]

    for i, q in enumerate(questions):
        img_path = image_dir / q["image"]
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        try:
            segs = segmenter.segment(img)
        except Exception as e:
            print(f"[{i}] segment failed: {e}"); continue
        if not segs or len(segs) < 2:
            continue
        enc = wrapper.prepare_inputs(img, q["text"])
        prompt_ids = wrapper.expand_image_tokens(enc["input_ids"])
        pixel_values = enc["pixel_values"]
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = wrapper.expand_image_tokens(attn_mask)

        # Pick the model's actual top-token answer to compute SHAP-H over
        out = wrapper.prefill(prompt_ids, pixel_values, attn_mask)
        first_tok = int(out.logits[0, -1, :].argmax().item())
        span = [first_tok]
        try:
            phis = shap_phis(wrapper, img, segs, prompt_ids, pixel_values,
                             attn_mask, span, max_batch=8)
            shap_H = _norm_entropy(phis)
        except Exception as e:
            print(f"[{i}] shap failed: {e}"); continue
        try:
            attn_H = _attn_H(wrapper, segs, prompt_ids, pixel_values, attn_mask)
        except Exception as e:
            print(f"[{i}] attn failed: {e}"); continue

        pred = "yes" if first_tok == yes_id else ("no" if first_tok == no_id else "?")
        rows.append((q.get("question_id", i), len(segs), shap_H, attn_H,
                     pred, q["label"]))
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(questions)}] shap_H={shap_H:.3f} attn_H={attn_H:.3f}")

    with open(args.out, "w") as f:
        f.write("qid,n_seg,shap_H,attn_H,pred,label\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    sh = np.array([r[2] for r in rows]); ah = np.array([r[3] for r in rows])
    mask = ~np.isnan(sh) & ~np.isnan(ah)
    sh, ah = sh[mask], ah[mask]
    corr = float(np.corrcoef(sh, ah)[0, 1]) if len(sh) > 1 else float("nan")
    print(f"\nN={len(sh)}  Pearson r = {corr:.3f}")
    print(f"SHAP-H mean={sh.mean():.3f} std={sh.std():.3f}")
    print(f"Attn-H mean={ah.mean():.3f} std={ah.std():.3f}")


if __name__ == "__main__":
    main()
