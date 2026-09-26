#!/usr/bin/env bash
# Full CHAIR eval: 4 methods × 2 protocols (greedy default + sampling).
#
# - baseline (greedy)        : SAE protocol main row
# - baseline_sample           : VCD protocol comparison
# - vcd (sampling=direct)     : VCD protocol
# - vcd_greedy (sampling=greedy in code) : SAE Tab. 1 "VCD on greedy" row
# - aif (greedy)              : already greedy
# - ours (greedy)             : already greedy
#
# 1000 images each (--n-images 1000).  Runs per GPU sequentially.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/chair_full logs

# We need to extend chair.py to also support baseline_sample and vcd_greedy.
# For now this script invokes whatever choices chair.py supports.

(
  for m in baseline vcd; do
    LOG="logs/chair_full_${m}.log"
    echo "[GPU0] >> CHAIR/$m  ($(date '+%F %T'))" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 python -m src.benchmarks.chair \
      --method "$m" --n-images 1000 \
      --out-dir results/chair_full 2>&1 | tee -a "$LOG"
  done
) &
GPU0_PID=$!

(
  for m in aif ours; do
    LOG="logs/chair_full_${m}.log"
    echo "[GPU1] >> CHAIR/$m  ($(date '+%F %T'))" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=1 python -m src.benchmarks.chair \
      --method "$m" --n-images 1000 \
      --out-dir results/chair_full 2>&1 | tee -a "$LOG"
  done
) &
GPU1_PID=$!

wait $GPU0_PID $GPU1_PID
echo "ALL CHAIR RUNS DONE  ($(date '+%F %T'))"
