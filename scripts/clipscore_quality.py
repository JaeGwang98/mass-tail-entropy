"""Rebuttal (R3-W3 follow-up): sentence-level CLIPScore on the CHAIR outputs
of every Table-3 method (semantic image-text consistency).

CLIPScore (Hessel et al., 2021): 2.5 * max(cos(E_img, E_text), 0), computed
per sentence (nltk sent_tokenize, CLIP 77-token truncation) and averaged per
caption, then over the 1000 captions. Sentence level avoids the 77-token
limit on 90-160-word captions and matches the per-claim granularity of
hallucination. Same raw_*.jsonl inputs as caption_quality.py; image
embeddings are cached (all methods share the same 1000 COCO images).

Usage: CUDA_VISIBLE_DEVICES=1 python scripts/clipscore_quality.py
"""
from __future__ import annotations

import json
from pathlib import Path

import nltk
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
COCO = ROOT / "data/coco/val2014"

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

CLIP_NAME = "openai/clip-vit-base-patch32"


@torch.no_grad()
def main():
    from transformers import CLIPModel, CLIPProcessor
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_NAME, torch_dtype=torch.float16).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_NAME)

    img_emb_cache: dict[str, torch.Tensor] = {}

    def embed_images(names):
        todo = [n for n in names if n not in img_emb_cache]
        for i in range(0, len(todo), 64):
            batch = todo[i:i + 64]
            ims = [Image.open(COCO / n).convert("RGB") for n in batch]
            px = proc(images=ims, return_tensors="pt").to(dev)
            e = model.get_image_features(pixel_values=px["pixel_values"].half())
            if not torch.is_tensor(e):          # transformers >= 5: ModelOutput
                e = e.pooler_output
            e = e / e.norm(dim=-1, keepdim=True)
            for n, v in zip(batch, e):
                img_emb_cache[n] = v
            if (i // 64) % 5 == 0:
                print(f"  img emb {i+len(batch)}/{len(todo)}", flush=True)

    def embed_texts(sents):
        out = []
        for i in range(0, len(sents), 256):
            tk = proc(text=sents[i:i + 256], return_tensors="pt",
                      padding=True, truncation=True, max_length=77).to(dev)
            e = model.get_text_features(input_ids=tk["input_ids"],
                                        attention_mask=tk["attention_mask"])
            if not torch.is_tensor(e):          # transformers >= 5: ModelOutput
                e = e.pooler_output
            out.append(e / e.norm(dim=-1, keepdim=True))
        return torch.cat(out)

    results = []
    for backbone, runs in RUNS.items():
        for name, path in runs.items():
            if not path.exists():
                print(f"!! missing {path}")
                continue
            recs = [json.loads(l) for l in open(path) if l.strip()]
            embed_images([r["image"] for r in recs])
            cap_scores = []
            sents_all, owner = [], []
            for j, r in enumerate(recs):
                ss = [s for s in nltk.sent_tokenize(r["caption"]) if s.strip()]
                sents_all.extend(ss)
                owner.extend([j] * len(ss))
            temb = embed_texts(sents_all)
            per_cap = [[] for _ in recs]
            for (j, e) in zip(owner, temb):
                ie = img_emb_cache[recs[j]["image"]]
                s = 2.5 * torch.clamp((ie * e).sum(), min=0).item()
                per_cap[j].append(s)
            cap_scores = [sum(v) / len(v) for v in per_cap if v]
            mean = sum(cap_scores) / len(cap_scores)
            results.append({"backbone": backbone, "method": name,
                            "n": len(cap_scores),
                            "clipscore_sent": mean})
            print(f"{backbone:10s} {name:9s} n={len(cap_scores):4d} "
                  f"CLIPScore-sent={mean:.4f}", flush=True)
    out = RES / "rebuttal_clipscore.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
