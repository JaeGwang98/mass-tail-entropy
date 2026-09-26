#!/usr/bin/env bash
# Run all 4 methods on POPE (random, popular, adversarial).
#
# Usage:
#   bash scripts/run_pope.sh [METHODS...] [-- extra args for the python script]
#
# Examples:
#   bash scripts/run_pope.sh                          # all four methods
#   bash scripts/run_pope.sh baseline vcd             # only those two
#   bash scripts/run_pope.sh -- --limit 200           # debug, all methods
set -euo pipefail
cd "$(dirname "$0")/.."

ALL=(baseline vcd aif ours)
METHODS=()
EXTRA=()
saw_dashdash=0
for arg in "$@"; do
  if [[ "$arg" == "--" ]]; then saw_dashdash=1; continue; fi
  if (( saw_dashdash )); then EXTRA+=("$arg"); else METHODS+=("$arg"); fi
done
[[ ${#METHODS[@]} -eq 0 ]] && METHODS=("${ALL[@]}")

mkdir -p results/pope logs
for m in "${METHODS[@]}"; do
  log="logs/pope_${m}.log"
  echo "==> POPE: ${m}  (log -> ${log})"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    python -m src.benchmarks.pope --method "${m}" --setting all \
      --out-dir results/pope "${EXTRA[@]}" 2>&1 | tee "${log}"
done
