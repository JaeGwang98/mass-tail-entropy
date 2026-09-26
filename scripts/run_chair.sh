#!/usr/bin/env bash
# Run all 4 methods on CHAIR.
#
# Usage:
#   bash scripts/run_chair.sh [METHODS...] [-- extra args]
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

mkdir -p results/chair logs
for m in "${METHODS[@]}"; do
  log="logs/chair_${m}.log"
  echo "==> CHAIR: ${m}  (log -> ${log})"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    python -m src.benchmarks.chair --method "${m}" \
      --out-dir results/chair "${EXTRA[@]}" 2>&1 | tee "${log}"
done
