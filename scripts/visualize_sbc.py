"""Qualitative visualization of SBC v3 — saves a PNG showing:
  - Image + segment overlay (top-K highlighted for MSB)
  - Per-segment SHAP φ bar chart + H value + routing badge
  - PMI logit blend (for PMI route) or top-K boost summary (for MSB route)
  - Baseline-greedy output vs SBC v3 output, with GT (POPE) / open-ended (CHAIR)

Two rows: one POPE example (PMI route preferred) + one CHAIR example (MSB
route). One PNG to ``results/sbc_v3_visualization.png``.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/visualize_sbc.py
    CUDA_VISIBLE_DEVICES=1 python scripts/visualize_sbc.py \
        --chair-image COCO_val2014_000000058350.jpg \
        --out results/sbc_v3_visualization_058350.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.llava_wrapper import LlavaWrapper                # noqa: E402
from src.utils.segmentation import PanopticSegmenter             # noqa: E402
from src.utils.common import load_config                         # noqa: E402
from src.decoding.baseline import greedy_decode                 # noqa: E402
from src.decoding.ours_sbc import (_lookahead_with_logp,        # noqa: E402
                                   _blank_span_matches,
                                   _norm_entropy, _generate_msb,
                                   _generate_pmi)
from src.decoding.ours_msb import _shap_phis_batched, _topk_diverse  # noqa: E402
from src.decoding.vcd import _apc_filter                        # noqa: E402
from src.benchmarks.pope import POPE_QUESTION_SUFFIX            # noqa: E402
from src.benchmarks.chair import CHAIR_PROMPT, sample_image_ids  # noqa: E402

TAU_MID = 0.5
ALPHA_PMI = 1.0
BETA = 0.1
BOOST = 1.5
TOP_K = 2


@torch.no_grad()
def run_sbc_instrumented(wrapper, segmenter, image, question, max_new_tokens):
    """Mirrors ours_sbc_decode but exposes intermediate state for viz."""
    enc = wrapper.prepare_inputs(image, question)
    input_ids = enc["input_ids"]
    pixel_v = enc["pixel_values"]
    attn = enc.get("attention_mask")

    segments = segmenter.segment(image, min_area_frac=0.01, max_segments=6)
    state = {"segments": segments}
    if len(segments) < 2:
        state["route"] = "fallback"
        state["output"] = greedy_decode(wrapper, image, question, max_new_tokens)
        return state

    span, base_lp, v_pl, v_pp = _lookahead_with_logp(
        wrapper, input_ids, pixel_v, attn, True, 8, max_steps=32)
    state["span"] = span
    if not span:
        state["route"] = "fallback"
        state["output"] = greedy_decode(wrapper, image, question, max_new_tokens)
        return state

    phis = _shap_phis_batched(wrapper, image, segments, input_ids, pixel_v,
                              attn, span, base_lp=base_lp)
    H = _norm_entropy(phis)
    state.update({"phis": phis, "H": H})

    if H >= TAU_MID:
        blank_match, b_pl, b_pp = _blank_span_matches(
            wrapper, input_ids, torch.zeros_like(pixel_v), attn, span,
            True, 8, max_steps=32)
    else:
        blank_match, b_pl, b_pp = False, None, None
    state["blank_match"] = bool(blank_match) if H >= TAU_MID else None

    if H >= TAU_MID and blank_match:
        # Capture first-token logits for the visualization
        logits_v = v_pl[0].float().detach().cpu().numpy()
        logits_b = b_pl[0].float().detach().cpu().numpy()
        keep_mask = _apc_filter(v_pl, BETA)[0].detach().cpu().numpy()
        blend = (1.0 + ALPHA_PMI) * logits_v - ALPHA_PMI * logits_b
        blend_apc = np.where(keep_mask, blend, -np.inf)
        out = _generate_pmi(wrapper, input_ids, pixel_v, attn, ALPHA_PMI, BETA,
                            max_new_tokens, v_prefill=(v_pl, v_pp),
                            b_prefill=(b_pl, b_pp))
        state.update({"route": "pmi", "alpha": ALPHA_PMI,
                      "logits_v": logits_v, "logits_b": logits_b,
                      "blend": blend_apc, "output": out})
    else:
        chosen = _topk_diverse(phis, segments, TOP_K, 0.5)
        out = _generate_msb(wrapper, input_ids, pixel_v, attn, segments, phis,
                            TOP_K, 0.5, BOOST, max_new_tokens)
        if out is None:
            out = greedy_decode(wrapper, image, question, max_new_tokens)
            state["route"] = "fallback"
        else:
            state["route"] = "msb"
        state.update({"chosen": chosen, "output": out})
    return state


def overlay_segments(image_pil, segments, chosen=None, alpha=0.45):
    arr = np.array(image_pil).astype(np.float32) / 255.0
    out = arr.copy()
    cmap = plt.get_cmap("tab10")
    for i, seg in enumerate(segments):
        rgb = np.array(cmap(i % 10)[:3])
        if chosen is not None and i not in chosen:
            rgb = rgb * 0.35 + 0.35              # muted gray tint
            a = alpha * 0.5
        else:
            a = alpha
        mask = seg.mask
        for ch in range(3):
            out[..., ch] = np.where(mask, a * rgb[ch] + (1 - a) * arr[..., ch],
                                    out[..., ch])
    # Draw thin borders for chosen segments
    if chosen is not None:
        for i in chosen:
            mask = segments[i].mask
            from scipy import ndimage
            try:
                edge = ndimage.binary_dilation(mask) ^ mask
                for ch in range(3):
                    out[..., ch] = np.where(edge, 1.0, out[..., ch])  # white edge
            except Exception:
                pass
    return (out * 255).clip(0, 255).astype(np.uint8)


def _bar_colors(K, chosen):
    cmap = plt.get_cmap("tab10")
    out = []
    for i in range(K):
        c = cmap(i % 10)
        if chosen is not None and i not in chosen:
            out.append((0.7, 0.7, 0.7, 1.0))
        else:
            out.append(c)
    return out


def draw_phi_bar(ax, phis, route_text, chosen=None):
    """Raw φ values (visually shows the absolute LOO drop per segment)."""
    K = len(phis)
    bars = ax.bar(range(K), phis, color=_bar_colors(K, chosen),
                  edgecolor='black', linewidth=0.6)
    ymax = max(abs(min(phis)), abs(max(phis)), 1e-6)
    for b, p in zip(bars, phis):
        y = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2,
                y + ymax * 0.04 if y >= 0 else y - ymax * 0.10,
                f"{p:+.2f}",
                ha='center', va='bottom' if y >= 0 else 'top',
                fontsize=9, family='monospace')
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"$s_{{{i}}}$" for i in range(K)])
    ax.set_ylabel(r"raw SHAP $\phi_i$ (LOO)")
    ax.set_title(f"raw $\\phi$ — absolute LOO drop  ({route_text})",
                 fontsize=11, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    ax.margins(y=0.22)


def draw_softmax_bar(ax, phis, H, chosen=None):
    """softmax(φ) — this is what H is computed on (the flatness signal SBC uses)."""
    K = len(phis)
    z = np.array(phis) - np.max(phis)
    sm = np.exp(z); sm /= sm.sum()
    bars = ax.bar(range(K), sm, color=_bar_colors(K, chosen),
                  edgecolor='black', linewidth=0.6)
    for b, p in zip(bars, sm):
        y = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, y + 0.015,
                f"{p:.2f}",
                ha='center', va='bottom',
                fontsize=9, family='monospace')
    uniform = 1.0 / K
    ax.axhline(uniform, color='red', linestyle=':', linewidth=1.1,
               label=f"uniform $1/K = {uniform:.2f}$")
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"$s_{{{i}}}$" for i in range(K)])
    ax.set_ylabel(r"softmax($\phi_i$)")
    regime = "flat" if H >= TAU_MID else "concentrated"
    ax.set_title(f"softmax($\\phi$) — $H = {H:.3f}$ ({regime})",
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(sm.max() * 1.30, uniform * 1.6))
    ax.grid(axis='y', alpha=0.3)


def draw_logit_panel(ax, logits_v, logits_b, blend, tokenizer, top=3):
    top_idx = np.argsort(-blend)[:top]
    # Add the runner-up from logits_v if not already included
    v_top = int(np.argmax(logits_v))
    if v_top not in top_idx:
        top_idx = np.concatenate([[v_top], top_idx])[:top + 1]
    labels = [tokenizer.decode([int(i)]).strip() or f"id={i}" for i in top_idx]
    x = np.arange(len(top_idx))
    w = 0.27
    ax.bar(x - w, logits_v[top_idx], w, label=r"$\log p(y|v)$",
           color='#1f77b4', edgecolor='black', linewidth=0.5)
    ax.bar(x, logits_b[top_idx], w, label=r"$\log p(y|v')$ (blank)",
           color='#ff7f0e', edgecolor='black', linewidth=0.5)
    ax.bar(x + w, blend[top_idx], w,
           label=r"$(1{+}\alpha)\,v - \alpha\,v'$", color='#2ca02c',
           edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"'{l}'" for l in labels])
    ax.set_ylabel("logit")
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title("PMI: prior-cancelled logit blend", fontsize=11, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)


def draw_text_panel(ax, q, gt, before, after, route):
    ax.axis('off')
    route_color = {"pmi": "#2ca02c", "msb": "#d62728", "fallback": "#7f7f7f"}.get(route, "#666666")
    # Wrap long captions
    def _wrap(s, w=60):
        if len(s) <= w:
            return s
        out = []
        while len(s) > w:
            sp = s.rfind(" ", 0, w)
            if sp <= 0:
                sp = w
            out.append(s[:sp])
            s = s[sp:].lstrip()
        out.append(s)
        return "\n  ".join(out)
    text = (
        f"Q: {q}\n"
        f"GT: {gt}\n\n"
        f"▶ Baseline (greedy):\n  {_wrap(before)!r}\n\n"
        f"▶ SBC v3 ({route}):\n  {_wrap(after)!r}"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#fafafa",
                      edgecolor=route_color, linewidth=1.5))


def find_pope_pmi_case_from_dispersed(segmenter, pope_dir, dispersed_path):
    """Use the recorded ``pope_dispersed_errors.json`` (PMI-recovery analysis)
    to pick a known case where SBC v3 routed PMI and corrected baseline.

    PMI predominantly flips false-negatives (baseline='No', gt='Yes')."""
    if not dispersed_path.exists():
        return None
    d = json.load(open(dispersed_path))
    for split in ("random", "popular", "adversarial"):
        for r in d[split]["rows"]:
            b = str(r["baseline"]).strip().lower()
            p = str(r["pmi"]).strip().lower()
            g = str(r["gt"]).strip().lower()
            if p != b and p == g:                       # PMI corrected to GT
                img_path = pope_dir / r["image"]
                if not img_path.exists():
                    continue
                img = Image.open(img_path).convert("RGB")
                segs = segmenter.segment(img, min_area_frac=0.01,
                                         max_segments=6)
                if len(segs) < 2:
                    continue
                full_q = r["question"] + POPE_QUESTION_SUFFIX
                # baseline output recorded in the json (saves a forward pass)
                return ({"image": r["image"], "text": r["question"],
                         "label": r["gt"]}, img, full_q, r["baseline"])
    return None


def find_chair_case(wrapper, segmenter, chair_dir, seed, n_try=10,
                    explicit_filename=None):
    """If ``explicit_filename`` is given, load that exact image. Otherwise
    return the first CHAIR sample with ≥2 segments."""
    if explicit_filename:
        fn = explicit_filename
        path = chair_dir / fn
        if not path.exists():
            raise FileNotFoundError(f"CHAIR image not found: {path}")
        cid_str = fn.split("_")[-1].split(".")[0]
        try:
            cid = int(cid_str)
        except ValueError:
            cid = -1
        img = Image.open(path).convert("RGB")
        return cid, fn, img
    for cid, fn in sample_image_ids(chair_dir, n_try, seed):
        img = Image.open(chair_dir / fn).convert("RGB")
        segs = segmenter.segment(img, min_area_frac=0.01, max_segments=6)
        if len(segs) < 2:
            continue
        return cid, fn, img
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chair-image", default=None,
                        help="Exact CHAIR filename (e.g. "
                             "COCO_val2014_000000058350.jpg). "
                             "Falls back to first viable sample if omitted.")
    parser.add_argument("--out", default="results/sbc_v3_visualization.png",
                        help="Output PNG path (relative to repo root or absolute)")
    args = parser.parse_args()

    cfg = load_config()
    w = LlavaWrapper(model_name=cfg["model"]["name"],
                     dtype=getattr(torch, cfg["model"]["dtype"]),
                     attn_implementation=cfg["model"]["attn_implementation"])
    seg = PanopticSegmenter(model_name=cfg["mask2former"]["name"],
                            dtype=getattr(torch, cfg["model"]["dtype"]))

    pope_dir = ROOT / cfg["benchmarks"]["pope"]["image_dir"]
    pope_qs = [json.loads(l) for l in open(
        ROOT / cfg["benchmarks"]["pope"]["data_dir"] / "coco_pope_adversarial.json")]
    chair_dir = ROOT / cfg["benchmarks"]["chair"]["image_dir"]

    print("== finding POPE PMI example via dispersed_errors.json ==",
          flush=True)
    dispersed_path = ROOT / "results" / "pope_dispersed_errors.json"
    pope = find_pope_pmi_case_from_dispersed(seg, pope_dir, dispersed_path)
    if pope is None:
        print("[POPE] no PMI-recovery case found; falling back to first viable")
        for q in pope_qs:
            img_path = pope_dir / q["image"]
            if not img_path.exists(): continue
            img = Image.open(img_path).convert("RGB")
            segs = seg.segment(img, min_area_frac=0.01, max_segments=6)
            if len(segs) < 2: continue
            full_q = q["text"] + POPE_QUESTION_SUFFIX
            base = greedy_decode(w, img, full_q, max_new_tokens=8)
            pope = (q, img, full_q, base); break
    pope_q, pope_img, pope_full_q, pope_baseline = pope
    print(f"[POPE] {pope_q['image']!r}  q={pope_q['text']!r}  "
          f"gt={pope_q.get('label')}  baseline={pope_baseline!r}", flush=True)

    print("== running SBC v3 on POPE ==", flush=True)
    pope_state = run_sbc_instrumented(w, seg, pope_img, pope_full_q,
                                      max_new_tokens=8)
    print(f"[POPE] route={pope_state['route']}  H={pope_state.get('H'):.3f}  "
          f"output={pope_state['output']!r}\n"
          f"       phi={np.array2string(pope_state['phis'], precision=3)}",
          flush=True)

    print(f"== finding CHAIR example "
          f"({'fixed='+args.chair_image if args.chair_image else 'auto'}) ==",
          flush=True)
    cid, fn, chair_img = find_chair_case(
        w, seg, chair_dir, cfg["benchmarks"]["chair"]["seed"],
        explicit_filename=args.chair_image)
    chair_baseline = greedy_decode(w, chair_img, CHAIR_PROMPT, max_new_tokens=128)
    print(f"[CHAIR] {fn!r}  baseline len={len(chair_baseline)}", flush=True)

    print("== running SBC v3 on CHAIR ==", flush=True)
    chair_state = run_sbc_instrumented(w, seg, chair_img, CHAIR_PROMPT,
                                       max_new_tokens=128)
    print(f"[CHAIR] route={chair_state['route']}  H={chair_state.get('H'):.3f}  "
          f"chosen={chair_state.get('chosen')}\n"
          f"        phi={np.array2string(chair_state['phis'], precision=3)}",
          flush=True)

    # ---- render ----
    fig, axes = plt.subplots(2, 5, figsize=(26, 11),
                             gridspec_kw={"width_ratios":
                                          [1.2, 0.95, 0.95, 1.2, 1.7]})

    # Row 1: POPE
    axes[0, 0].imshow(overlay_segments(pope_img, pope_state["segments"]))
    axes[0, 0].set_title(f"Image + segments "
                         f"(K={len(pope_state['segments'])})",
                         fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    badge = f"{pope_state['route'].upper()}"
    if pope_state["route"] == "pmi" and pope_state.get("blank_match"):
        badge += "  (blank-agreement ✓)"
    draw_phi_bar(axes[0, 1], pope_state["phis"], badge)
    draw_softmax_bar(axes[0, 2], pope_state["phis"], pope_state["H"])
    if pope_state["route"] == "pmi":
        draw_logit_panel(axes[0, 3], pope_state["logits_v"],
                         pope_state["logits_b"], pope_state["blend"],
                         w.tokenizer)
    else:
        axes[0, 3].axis('off')
        axes[0, 3].text(0.5, 0.5,
                        f"route = {pope_state['route']}\n"
                        f"(H < $\\tau_{{mid}}$ OR blank-disagree)",
                        ha='center', va='center', fontsize=11,
                        transform=axes[0, 3].transAxes)
    draw_text_panel(axes[0, 4], pope_q["text"], pope_q.get("label"),
                    pope_baseline, pope_state["output"], pope_state["route"])

    # Row 2: CHAIR
    chosen = chair_state.get("chosen")
    axes[1, 0].imshow(overlay_segments(chair_img, chair_state["segments"],
                                       chosen=chosen))
    axes[1, 0].set_title(f"Image + top-{TOP_K} boosted segments",
                         fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    badge = (f"{chair_state['route'].upper()}"
             + (f"  top-{TOP_K} boost γ={BOOST}" if chair_state["route"] == "msb"
                else ""))
    draw_phi_bar(axes[1, 1], chair_state["phis"], badge, chosen=chosen)
    draw_softmax_bar(axes[1, 2], chair_state["phis"], chair_state["H"],
                     chosen=chosen)
    axes[1, 3].axis('off')
    if chair_state["route"] == "msb" and chosen is not None:
        chosen_str = ", ".join(f"$s_{{{i}}}$ ($\\phi$={chair_state['phis'][i]:.2f})"
                               for i in chosen)
        axes[1, 3].text(0.5, 0.5,
                        f"MSB attention boost\n\n"
                        f"chosen = {{{chosen_str}}}\n\n"
                        f"4-D attn mask:\n"
                        f"$\\text{{score}}[t \\to v_j] +\\!\\!= \\log\\gamma$\n"
                        f"for $v_j \\in$ chosen segments",
                        ha='center', va='center', fontsize=10.5,
                        transform=axes[1, 3].transAxes,
                        bbox=dict(boxstyle="round,pad=0.6",
                                  facecolor="#fff3cd",
                                  edgecolor="#e6a23c", linewidth=1.3))
    draw_text_panel(axes[1, 4], "Please describe this image in detail.",
                    "(open-ended)", chair_baseline, chair_state["output"],
                    chair_state["route"])

    fig.suptitle("SBC v3 — SHAP-Guided Bimodal Calibration  (qualitative)",
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f"\n✓ saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
