#!/usr/bin/env bash
# Full POPE evaluation: 4 methods × 2 protocols (VCD sampling, AIF greedy).
# - baseline (sampling 5 runs) for VCD protocol
# - baseline_greedy for AIF protocol
# - vcd (sampling 5 runs) for VCD protocol
# - vcd_greedy for AIF protocol
# - aif and ours are deterministic — single run, used for both protocols
#
# Layout uses 2 GPUs in parallel.
#   GPU 0 :  baseline (sample 5x) -> vcd (sample 5x) -> baseline_greedy -> vcd_greedy
#   GPU 1 :  aif -> ours
#
# Output goes to results/pope_full/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/pope_full logs

(
  for m in baseline vcd baseline_greedy vcd_greedy; do
    LOG="logs/pope_full_${m}.log"
    echo "[GPU0] >> $m  ($(date '+%F %T'))" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 python -m src.benchmarks.pope \
      --method "$m" --setting all \
      --out-dir results/pope_full 2>&1 | tee -a "$LOG"
  done
) &
GPU0_PID=$!

(
  for m in aif ours; do
    LOG="logs/pope_full_${m}.log"
    echo "[GPU1] >> $m  ($(date '+%F %T'))" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=1 python -m src.benchmarks.pope \
      --method "$m" --setting all \
      --out-dir results/pope_full 2>&1 | tee -a "$LOG"
  done
) &
GPU1_PID=$!

wait $GPU0_PID $GPU1_PID
echo "ALL POPE RUNS DONE  ($(date '+%F %T'))"
