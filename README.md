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

`configs/default.yaml` holds the paper configuration and is loaded by every
entry point; pass another file with `--config` (e.g. the $K$ ablation below).

```bash
# Main tables (Tables for CHAIR / POPE / MME): SBC rows = --method ours_sbc
python -m src.benchmarks.chair --method ours_sbc --n-images 1000 \
  --model llava-hf/llava-1.5-7b-hf                    # CHAIR, 1000 COCO images
python -m src.benchmarks.pope  --method ours_sbc --setting all \
  --model llava-hf/llava-1.5-7b-hf                    # POPE, 3 splits x 3000 q
python -m src.benchmarks.mme   --method ours_sbc \
  --model llava-hf/llava-1.5-7b-hf                    # MME hallucination subset
```

Use `--model Qwen/Qwen2.5-VL-7B-Instruct` for the Qwen2.5-VL rows. All runs
use one shared configuration with no per-backbone tuning.

## Method names

The paper's main-table SBC rows were produced with the **three-condition
router** (`ours_sbc`, which also evaluates $H \ge \tau$ before routing). The
two-signal router described in the paper, which drops that condition and runs
segmentation only on the MSB route, is `ours_lazy`; it gives the
`no_h` column of the gate-ablation table and the "lazy" rows of the
cost table.

| `--method`          | Paper                                                            |
|---------------------|------------------------------------------------------------------|
| `baseline_greedy` (POPE, MME) / `baseline` (CHAIR) | Greedy baseline. In POPE/MME, `baseline` is the *sampling* baseline (MME sampling-protocol table) |
| `vcd`               | VCD with direct sampling as in the VCD paper (CHAIR, POPE; POPE averaged over 5 runs) |
| `vcd_greedy`        | VCD under greedy decoding (MME table)                            |
| `opera`             | OPERA, greedy adaptation: over-trust penalty only, beam retrospection omitted |
| `aif`               | AIF, reproduced from the paper (selected mask ratio capped at 0.5) |
| `pai`               | PAI                                                              |
| `ours_msb_sent`     | MSB-only                                                         |
| `ours_pmi`          | PMI-only                                                         |
| `ours_sbc`          | **SBC, main tables** (three-condition router)                    |
| `ours_lazy`         | SBC, two-signal router (no $H$ test; segmentation only on the MSB route) |
| `ours_lazy_attn`    | as `ours_lazy`, MSB ranks segments by attention (one extra attention prefill, no occlusion passes) |
| `ours_no_h`         | three-condition router with the $H$ condition removed (POPE, MME) |
| `ours_logit_h` / `ours_attn_h` / `ours_task_pmi` / `ours_task_msb` | gate-substitution ablations (POPE). `ours_task_pmi` applies the same rule as `ours_no_h` |

Not every method is exposed by every benchmark script; run
`python -m src.benchmarks.<name> -h` for the exact list.

## Reproducing the paper's analyses

All commands run from the repository root and write under `results/`.

```bash
# H distributions (mass-tail table, further backbones, SAM segmenter)
python scripts/h_distribution_anymodel.py --model llava-hf/llava-1.5-7b-hf \
  --out results/h_dist_llava7b.json                   # add --segmenter sam, --max-segments 4|8,
                                                      # --load-8bit for LLaVA-1.5-13B
python scripts/sbc_h_distribution.py 200              # SBC routing / H on POPE (200 q per split)
python scripts/amber_h_distribution.py --out results/amber_h_dist_llava7b.json

# Exact Shapley vs LOO (all 2^K coalitions)
python -m scripts.diag_exactshap --bench pope --setting adversarial --limit 240
python -m scripts.diag_exactshap --bench chair --limit 240
python -m scripts.compare_loo_exact results/exactshap/<cell_dir>

# Generation-quality proxies on saved CHAIR outputs (edit RUNS at the top of
# each script to point at your raw_*.jsonl files)
python scripts/caption_quality.py                     # length, distinct-n, repetition (CPU)
python scripts/clipscore_quality.py                   # sentence-level CLIPScore

# Latency (LLaVA-1.5-7B)
python scripts/measure_latency.py --n 50              # POPE, s/question
python scripts/measure_latency_chair.py --n 20        # CHAIR, s/image (incl. ours_lazy, ours_lazy_attn)

# AMBER
python scripts/run_amber.py --task gen  --method ours_sbc --out-dir results/amber_gen_sbc
python scripts/run_amber.py --task disc --method ours_sbc --disc-file existence \
  --out-dir results/amber_disc_sbc                    # --disc-file attribute for the balanced subset

# Segment-count ablation (K = 4 / 8)
python -m src.benchmarks.pope --method ours_sbc --setting random --limit 1000 \
  --config configs/sbc_k4.yaml
```

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
