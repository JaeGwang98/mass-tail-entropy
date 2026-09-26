"""Generate the final consolidated tables for POPE + CHAIR across both
protocols (VCD = sampling, AIF = greedy), plus the v4 pilot.

Usage:
    python scripts/final_report.py
"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POPE_DIR = ROOT / "results" / "pope_full"
CHAIR_DIR = ROOT / "results" / "chair_full"
PILOT_DIR = ROOT / "results" / "pilot_v4"


def fmt_run(d, key):
    a = d["aggregate"][key]
    return f"{a['mean']*100:5.2f}±{a['std']*100:.2f}"


def avg_acc(d):
    accs = [v["aggregate"]["accuracy"]["mean"] for v in d.values()]
    return sum(accs) / len(accs) * 100


def load(path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def print_pope_vcd_protocol():
    """VCD protocol: per-split accuracy/precision/recall/F1/yes-ratio with
    standard deviation across 5 sampling runs (where applicable)."""
    print("\n" + "=" * 88)
    print("Table 1.  POPE  —  VCD protocol (sampling, 5 runs averaged where applicable)")
    print("=" * 88)
    print(f"{'method':<10} {'split':<12} {'acc':>12} {'prec':>12} {'rec':>12} {'F1':>12} {'yes%':>12}")
    for m in ["baseline", "vcd", "aif", "ours"]:
        d = load(POPE_DIR / f"summary_{m}.json")
        if d is None:
            continue
        for split in ["random", "popular", "adversarial"]:
            v = d.get(split)
            if v is None:
                continue
            print(f"{m:<10} {split:<12} {fmt_run(v, 'accuracy'):>12} "
                  f"{fmt_run(v, 'precision'):>12} {fmt_run(v, 'recall'):>12} "
                  f"{fmt_run(v, 'f1'):>12} {fmt_run(v, 'yes_ratio'):>12}")


def print_pope_aif_protocol():
    """AIF protocol: averaged accuracy across 3 splits (greedy, 1 run)."""
    print("\n" + "=" * 70)
    print("Table 2.  POPE  —  AIF protocol (greedy, avg-acc across 3 splits)")
    print("=" * 70)
    print(f"{'method':<22} {'random':>10} {'popular':>10} {'adv':>10} {'avg-acc':>10}")
    for m in ["baseline_greedy", "vcd_greedy", "aif", "ours"]:
        d = load(POPE_DIR / f"summary_{m}.json")
        if d is None:
            continue
        accs = [d[s]["aggregate"]["accuracy"]["mean"] for s in ["random", "popular", "adversarial"]
                if s in d]
        if len(accs) != 3:
            continue
        print(f"{m:<22} {accs[0]*100:>10.2f} {accs[1]*100:>10.2f} {accs[2]*100:>10.2f} "
              f"{sum(accs)/3*100:>10.2f}")


def print_chair():
    """CHAIR: SAE protocol (greedy) primary, with sampling for VCD comparison."""
    print("\n" + "=" * 76)
    print("Table 3.  CHAIR  —  greedy decoding (SAE protocol), 1000 random val imgs")
    print("=" * 76)
    print(f"{'method':<20} {'CS↓':>8} {'CI↓':>8} {'recall':>8} {'avg_len':>8}")
    for m in ["baseline", "vcd", "aif", "ours"]:
        d = load(CHAIR_DIR / f"summary_{m}.json")
        if d is None:
            continue
        print(f"{m:<20} {d['chair_s']*100:>8.2f} {d['chair_i']*100:>8.2f} "
              f"{d['recall']*100:>8.2f} {d['avg_len']:>8.1f}")


def print_pilot():
    summary = load(PILOT_DIR / "summary.json")
    supp = load(PILOT_DIR / "supplement.json") or {"pope": [], "chair": []}
    if summary is None:
        return
    print("\n" + "=" * 92)
    print("Table 4.  Pilot v4  —  ours-v4 boost-factor sweep on 200q POPE + 16imgs CHAIR (greedy)")
    print("=" * 92)
    pope_rows = {r["method"]: r for r in summary["pope"] + supp["pope"]}
    chair_rows = {r["method"]: r for r in summary["chair"] + supp["chair"]}
    pope_order = ["baseline_greedy", "vcd_greedy", "aif", "ours-v3",
                  "ours-v4 1.5x", "ours-v4 2.0x", "ours-v4 3.0x", "ours-v4 4.0x"]
    chair_order = ["baseline_greedy", "vcd_greedy", "aif", "ours-v3",
                   "ours-v4 2.0x", "ours-v4 3.0x"]
    print("--- POPE ---")
    print(f"{'method':<22} {'acc':>8} {'F1':>8} {'yes%':>8}")
    for n in pope_order:
        r = pope_rows.get(n)
        if r is None:
            continue
        print(f"{n:<22} {r['acc']*100:>8.2f} {r['f1']*100:>8.2f} {r['yes']*100:>8.2f}")
    print("\n--- CHAIR ---")
    print(f"{'method':<22} {'CS↓':>8} {'CI↓':>8} {'recall':>8} {'avg_len':>8}")
    for n in chair_order:
        r = chair_rows.get(n)
        if r is None:
            continue
        print(f"{n:<22} {r['cs']*100:>8.2f} {r['ci']*100:>8.2f} "
              f"{r['recall']*100:>8.2f} {r['avg_len']:>8.1f}")


def main():
    print_pope_vcd_protocol()
    print_pope_aif_protocol()
    print_chair()
    print_pilot()


if __name__ == "__main__":
    main()
