"""Regression diff: refactored SBC vs pre-refactor known-good raw outputs.

PASS criterion: byte-identical predictions (POPE) and captions (CHAIR) on the
overlapping subset. Any mismatch => refactor changed model behaviour.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p, key):
    out = {}
    for line in open(p):
        r = json.loads(line)
        out[key(r)] = r
    return out


def main():
    fails = 0
    pope_dir = sys.argv[1] if len(sys.argv) > 1 else "results/regr_pope"
    chair_dir = sys.argv[2] if len(sys.argv) > 2 else "results/regr_chair"

    # POPE: match by (image, question), compare pred
    new = load(ROOT / pope_dir / "raw_ours_sbc_random_run0.jsonl",
               lambda r: (r["image"], r["question"]))
    old = load(ROOT / "results/pope_sbc_full_B/raw_ours_sbc_random_run0.jsonl",
               lambda r: (r["image"], r["question"]))
    common = [k for k in new if k in old]
    pope_mismatch = [k for k in common if new[k]["pred"] != old[k]["pred"]]
    print(f"POPE: compared {len(common)} q  |  pred mismatches: "
          f"{len(pope_mismatch)}")
    for k in pope_mismatch[:5]:
        print(f"  {k}: new={new[k]['pred']!r} old={old[k]['pred']!r}")
    fails += len(pope_mismatch)

    # CHAIR: match by image_id, compare caption byte-for-byte
    newc = load(ROOT / chair_dir / "raw_ours_sbc.jsonl",
                lambda r: r["image_id"])
    oldc = load(ROOT / "results/chair_sbc_full/raw_ours_sbc.jsonl",
                lambda r: r["image_id"])
    commonc = [k for k in newc if k in oldc]
    chair_mismatch = [k for k in commonc
                      if newc[k]["caption"] != oldc[k]["caption"]]
    print(f"CHAIR: compared {len(commonc)} imgs  |  caption mismatches: "
          f"{len(chair_mismatch)}")
    for k in chair_mismatch[:3]:
        print(f"  img {k}:")
        print(f"    new: {newc[k]['caption'][:160]!r}")
        print(f"    old: {oldc[k]['caption'][:160]!r}")
    fails += len(chair_mismatch)

    print()
    if fails == 0:
        print("PASS — refactor is byte-identical to pre-refactor SBC on "
              f"{len(common)} POPE q + {len(commonc)} CHAIR imgs")
        sys.exit(0)
    print(f"FAIL — {fails} mismatches; refactor changed behaviour")
    sys.exit(1)


if __name__ == "__main__":
    main()
