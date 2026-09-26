"""Write the consolidated results to results/RESULTS.md.

Includes baseline / VCD / AIF / ours-v3 on POPE (both protocols) and CHAIR.
"""

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POPE_DIR = ROOT / "results" / "pope_full"
CHAIR_DIR = ROOT / "results" / "chair_full"

DISPLAY = {
    "baseline":        "Baseline (sampling)",
    "vcd":             "VCD",
    "baseline_greedy": "Baseline (greedy)",
    "vcd_greedy":      "VCD",
    "aif":             "AIF",
    "ours":            "Ours (v3)",
}


def load(p):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def md_metric(d, key):
    a = d["aggregate"][key]
    return f"{a['mean']*100:.2f} ± {a['std']*100:.2f}"


def header():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return [
        "# Hallucination Mitigation — Final Results",
        "",
        f"*Generated: {now}*",
        "",
        "Model: **LLaVA-1.5-7B** (`llava-hf/llava-1.5-7b-hf`)  ",
        "Mask2Former: facebook/mask2former-swin-large-coco-panoptic",
        "",
        "Methods compared:",
        "- **Baseline** — regular decoding",
        "- **VCD** — Visual Contrastive Decoding (Leng et al., CVPR 2024)",
        "- **AIF** — Adaptive Information Flow (Liu et al., 2026)",
        "- **Ours (v3)** — SHAP-Targeted Visual Contrastive Decoding (방법론.md)",
        "",
        "Two evaluation protocols:",
        "- **VCD protocol**: direct sampling, 5 runs averaged. Each question is sampled `n=5` times and metrics are averaged with std-dev. Used by VCD paper Tab. 1.",
        "- **AIF protocol**: greedy decoding, single run. Reports averaged accuracy across 3 splits. Used by AIF paper Tab. 4.",
        "",
        "POPE prompt: `<question> Please answer this question with one word.`  ",
        "CHAIR prompt: `Please describe this image in detail.` (max_new_tokens=512)",
        "",
    ]


def section_pope_vcd():
    lines = [
        "## Table 1. POPE — VCD protocol (sampling, 5 runs)",
        "",
        "MSCOCO POPE; 500 images × 6 questions per (split, sampling-run) = 3000 questions/split, 5 runs.",
        "Numbers are mean ± std across the 5 runs.",
        "",
        "| Method | Split | Accuracy | Precision | Recall | F1 | Yes-ratio |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for m in ["baseline", "vcd", "aif", "ours"]:
        d = load(POPE_DIR / f"summary_{m}.json")
        if d is None:
            lines.append(f"| {DISPLAY[m]} | (pending) | — | — | — | — | — |")
            continue
        for split in ["random", "popular", "adversarial"]:
            v = d.get(split)
            if v is None:
                continue
            lines.append(
                f"| {DISPLAY[m]} | {split} | "
                f"{md_metric(v, 'accuracy')} | {md_metric(v, 'precision')} | "
                f"{md_metric(v, 'recall')} | {md_metric(v, 'f1')} | "
                f"{md_metric(v, 'yes_ratio')} |"
            )
    lines.append("")
    return lines


def section_pope_aif():
    lines = [
        "## Table 2. POPE — AIF protocol (greedy, 1 run)",
        "",
        "MSCOCO POPE; greedy decoding, single deterministic run.",
        "Reported metric: accuracy per split + averaged accuracy across 3 splits.",
        "",
        "| Method | Random | Popular | Adversarial | **Avg-Acc** |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in ["baseline_greedy", "vcd_greedy", "aif", "ours"]:
        d = load(POPE_DIR / f"summary_{m}.json")
        if d is None:
            lines.append(f"| {DISPLAY[m]} | — | — | — | — |")
            continue
        accs = []
        for s in ["random", "popular", "adversarial"]:
            v = d.get(s)
            if v is None:
                accs.append(None)
            else:
                accs.append(v["aggregate"]["accuracy"]["mean"])
        if any(a is None for a in accs):
            lines.append(f"| {DISPLAY[m]} | — | — | — | — |")
            continue
        avg = sum(accs) / 3
        lines.append(f"| {DISPLAY[m]} | {accs[0]*100:.2f} | {accs[1]*100:.2f} | "
                     f"{accs[2]*100:.2f} | **{avg*100:.2f}** |")
    lines.append("")
    return lines


def section_chair():
    lines = [
        "## Table 3. CHAIR — greedy (SAE protocol, 1000 random val2014 imgs)",
        "",
        "Free-form caption hallucination evaluation.  Lower CS / CI = better.",
        "",
        "| Method | CS ↓ | CI ↓ | Recall ↑ | Avg len |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in ["baseline", "vcd", "aif", "ours"]:
        d = load(CHAIR_DIR / f"summary_{m}.json")
        if d is None:
            lines.append(f"| {DISPLAY[m]} | — | — | — | — |")
            continue
        lines.append(
            f"| {DISPLAY[m]} | {d['chair_s']*100:.2f} | {d['chair_i']*100:.2f} | "
            f"{d['recall']*100:.2f} | {d['avg_len']:.1f} |"
        )
    lines.append("")
    return lines


def section_paper_reference():
    return [
        "## Reference (paper-reported numbers)",
        "",
        "**VCD paper, LLaVA-1.5-7B, MSCOCO POPE (sampling, 5 runs)**",
        "",
        "| Method | Split | Accuracy | F1 | Yes-ratio |",
        "|---|---|---:|---:|---:|",
        "| Baseline | random | 83.29 ± 0.35 | 81.33 ± 0.41 | 39.49 |",
        "| Baseline | popular | 81.88 ± 0.48 | 80.06 ± 0.05 | 40.92 |",
        "| Baseline | adversarial | 78.96 ± 0.52 | 77.57 ± 0.57 | 43.81 |",
        "| VCD | random | 87.73 ± 0.40 | 87.16 ± 0.41 | 45.55 |",
        "| VCD | popular | 85.38 ± 0.38 | 85.06 ± 0.37 | 47.91 |",
        "| VCD | adversarial | 80.88 ± 0.33 | 81.33 ± 0.34 | 52.42 |",
        "",
        "**AIF paper, LLaVA-1.5-7B, COCO POPE (greedy, 3-split avg-acc)**",
        "",
        "| Method | Avg-Acc |",
        "|---|---:|",
        "| Baseline | 85.4 |",
        "| AIF | 88.7 |",
        "",
        "**SAE paper, LLaVA-1.5-7B, CHAIR (greedy, 1000 imgs)**",
        "",
        "| Method | CS ↓ | CI ↓ |",
        "|---|---:|---:|",
        "| Baseline (greedy) | 47.9 | 13.6 |",
        "| Beam search | 52.1 | 14.1 |",
        "| Nucleus sampling | 56.0 | 16.4 |",
        "| OPERA | 48.5 | 13.7 |",
        "| VCD (greedy) | 55.4 | 15.7 |",
        "| PAI | 30.1 | 8.8 |",
        "| VAR | 25.3 | 6.5 |",
        "",
    ]


def main():
    parts = []
    parts += header()
    parts += section_pope_vcd()
    parts += section_pope_aif()
    parts += section_chair()
    parts += section_paper_reference()

    out = ROOT / "results" / "RESULTS.md"
    out.write_text("\n".join(parts))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
