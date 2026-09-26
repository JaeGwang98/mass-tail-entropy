# Mass-Tail Attribution Entropy: A Regime-Conditional Decoder for Vision-Language Model Hallucination

Official code for the Findings of AACL-IJCNLP 2026 paper
**"Mass-Tail Attribution Entropy: A Regime-Conditional Decoder for
Vision-Language Model Hallucination"** (Jaegwang Shin, Suan Lee).

The repository contains two things:

1. **The regime diagnostic.** The normalised entropy $H$ of leave-one-out (LOO)
   occlusion attributions over panoptic image segments. A VLM's inference
   mass, correct and hallucinated outputs alike, concentrates in one
   task-conditional tail of $H$. Captioning falls in the *over-concentration*
   tail (low $H$) and binary QA in the *over-spread* tail (high $H$).
2. **SBC (Segment-Based Conditioning).** A training-free decoder that routes
   between two interventions without computing attributions for the router:

   | Regime               | Intervention                                              |
   |----------------------|-----------------------------------------------------------|
   | over-concentration   | **MSB**: additive attention boost on the top-ranked segments |
   | over-spread          | **PMI**: blank-image contrastive logit blend              |
   | neither              | greedy                                                    |

   The router uses two cheap signals: whether the blank-image continuation
   agrees with the on-image one, and the on-image token margin (guard $\delta$).
   The diagnostic runs **once per generated response** (first-sentence
   lookahead), not per token. Segmentation and the $K$ occlusion passes run
   only on the MSB route.

## Layout

```
configs/
  default.yaml                  shared hyperparameters (paper defaults)
  sbc_k4.yaml / sbc_k8.yaml     segment-count ablation (K = 4 / 8)
src/
  models/                       LLaVA-1.5 / Qwen2-VL / Qwen2.5-VL wrappers
  utils/segmentation.py         Mask2Former panoptic segmentation + mean-colour occlusion
  decoding/
    ours_sbc.py                 SBC (all router variants, see --method below)
    ours_msb.py                 MSB actuator / MSB-only ablation
    ours_pmi.py                 PMI actuator / PMI-only ablation
    exact_shap.py               exact Shapley values over all 2^K coalitions
    ours_v3.py, ours_v4.py,
    ours_msb_rolling.py         shared lookahead / segment-to-token helpers
    baseline.py vcd.py opera.py aif.py pai.py   baselines
  benchmarks/                   CHAIR, POPE, MME, AMBER evaluation entry points
scripts/                        analysis, diagnostics, latency and figure scripts
```

## Setup

```bash
pip install -r requirements.txt
```

Expected data layout (not included in this repository):

```
data/coco/val2014/        COCO val2014 images (CHAIR, POPE)
data/coco/annotations/    COCO instance / caption annotations (CHAIR)
data/POPE/                POPE random / popular / adversarial splits
data/AMBER/               AMBER benchmark
```

A single 24 GB GPU (RTX 3090 / 4090) is enough for every 7B run (fp16).

## Running

```bash
# POPE (3 splits x 3000 questions)
python -m src.benchmarks.pope  --method ours_lazy --setting all \
  --model llava-hf/llava-1.5-7b-hf

# CHAIR (1000 COCO val2014 images)
python -m src.benchmarks.chair --method ours_lazy --n-images 1000 \
  --model llava-hf/llava-1.5-7b-hf

# MME hallucination subset
python -m src.benchmarks.mme   --method ours_lazy \
  --model llava-hf/llava-1.5-7b-hf
```

Use `--model Qwen/Qwen2.5-VL-7B-Instruct` for the Qwen2.5-VL rows. All runs
use one shared configuration (`configs/default.yaml`) with no per-backbone
tuning.

## Method names

| `--method`          | Paper                                                            |
|---------------------|------------------------------------------------------------------|
| `baseline_greedy` (POPE) / `baseline` (CHAIR, MME) | Greedy baseline                |
| `vcd` / `vcd_greedy`| VCD                                                              |
| `opera`             | OPERA                                                            |
| `aif`               | AIF                                                              |
| `pai`               | PAI                                                              |
| `ours_msb_sent`     | MSB-only                                                         |
| `ours_pmi`          | PMI-only                                                         |
| `ours_lazy`         | **SBC**: two-signal router, segmentation deferred to the MSB route |
| `ours_lazy_attn`    | SBC with attention-ranked MSB (no occlusion passes)              |
| `ours_sbc`          | SBC with the original three-condition router (incl. $H \ge \tau$) |
| `ours_no_h`         | three-condition router with the $H$ condition removed (POPE, MME) |
| `ours_logit_h` / `ours_attn_h` / `ours_task_pmi` / `ours_task_msb` | gate-substitution ablations (POPE) |

Not every method is exposed by every benchmark script; run
`python -m src.benchmarks.<name> -h` for the exact list.

## Hyperparameters (paper defaults)

| Param              | Value | Role                                      |
|--------------------|-------|-------------------------------------------|
| max segments $K$   | 6     | Mask2Former panoptic segments             |
| MSB top-$k$        | 2     | segments boosted by MSB                   |
| $b$                | 1.8   | MSB boost factor                          |
| PMI $\alpha$, $\beta$ | 1.0, 0.1 | VCD defaults (not tuned)             |
| $\delta$           | 0.5   | on-image margin guard                     |
| $\tau_{\mathrm{mid}}$ | 0.5 | $H$ threshold (three-condition router only) |

## Citation

```bibtex
@inproceedings{shin2026masstail,
  title     = {Mass-Tail Attribution Entropy: A Regime-Conditional Decoder
               for Vision-Language Model Hallucination},
  author    = {Shin, Jaegwang and Lee, Suan},
  booktitle = {Findings of the Association for Computational Linguistics:
               AACL-IJCNLP 2026},
  year      = {2026}
}
```

## Acknowledgments

This work was supported by the Ministry of Science and ICT and the National
Research Foundation of Korea (NRF) grant funded by the Korean government
(MSIT) (No. RS-2026-25498341).

## License

MIT (see `LICENSE`). Baseline implementations follow their original papers;
please also cite them if you use those components.
