#!/usr/bin/env bash
# InternVL3-8B third-family DECODING queue (camera-ready, reviewer R2).
# Same protocol as the paper's main tables (Qwen2.5-VL rows produced by
# scripts/gpu1_master.sh): baseline greedy vs SBC (gate v3, top-k=2, b=1.8,
# PMI alpha=1.0 / beta=0.1, delta=0.5, tau_mid=0.5, Mask2Former K<=6,
# sentence lookahead <=32 tokens). Tiling disabled (one 448x448 view).
#
# GPU 1 ONLY. Sequential, resumable (a stage is skipped when its output
# summary exists), one retry per stage. Before every stage it waits until
# GPU 1 has no foreign compute process and >= MIN_FREE_MB free memory.
# Pause between stages by creating logs/internvl_decoding.PAUSE.
#
# Launch:
#   setsid nohup bash scripts/internvl_decoding_queue.sh \
#       < /dev/null > logs/internvl_decoding_queue.out 2>&1 &
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1          # never touch GPU 0
export PYTHONUNBUFFERED=1

PY="python"
MODEL="OpenGVLab/InternVL3-8B-hf"
GPU_INDEX=1
MIN_FREE_MB=22000
STATUS="logs/internvl_decoding_status.txt"
QLOG="logs/internvl_decoding_queue.log"
PAUSE="logs/internvl_decoding.PAUSE"
LOCK="/tmp/shap_vlm_internvl_decoding_gpu1.lock"
SBC_ARGS=(--boost-factor 1.8 --image-margin-thresh 0.5)
mkdir -p logs results

exec 9>"$LOCK"
if ! flock -n 9; then echo "another internvl queue holds $LOCK; exit"; exit 0; fi

log()    { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }
status() { echo "$(date '+%F %T') $1 ${2:-}" >> "$STATUS"; log "STATUS $1 ${2:-}"; }

gpu1_ready() {
  local uuid free n
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU_INDEX" | tr -d ' ')
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_INDEX" | tr -d ' ')
  n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
      | tr -d ' ' | grep -c "^${uuid},")
  [[ -n "$free" && "$free" -ge "$MIN_FREE_MB" && "$n" -eq 0 ]]
}

wait_turn() {
  local warned=0
  while [[ -e "$PAUSE" ]] || ! gpu1_ready; do
    if [[ $warned -eq 0 ]]; then
      if [[ -e "$PAUSE" ]]; then status PAUSED "$PAUSE exists"
      else
        status WAITING "GPU1 busy (foreign compute proc or <${MIN_FREE_MB}MB free): $(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')"
      fi
      warned=1
    fi
    sleep 60
  done
}

# run_stage NAME DONE_FILE LOGFILE -- cmd...
run_stage() {
  local name="$1" done_file="$2" logf="$3"; shift 3
  if [[ -s "$done_file" ]]; then status SKIP "$name (exists: $done_file)"; return 0; fi
  local attempt
  for attempt in 1 2; do
    wait_turn
    status START "$name attempt=$attempt log=$logf"
    local t0=$SECONDS
    CUDA_VISIBLE_DEVICES=1 "$@" >> "$logf" 2>&1
    local rc=$?
    if [[ $rc -eq 0 && -s "$done_file" ]]; then
      status DONE "$name ($((SECONDS - t0))s) -> $done_file"
      return 0
    fi
    status FAIL "$name attempt=$attempt rc=$rc ($((SECONDS - t0))s); see $logf"
    sleep 60
  done
  return 1
}

status QUEUE_START "pid=$$ model=$MODEL"

# ---------------- MME (240 q) ----------------
run_stage mme_baseline results/mme_baseline_greedy_internvl3_8b/summary_baseline_greedy.json \
  logs/internvl_mme_baseline.log \
  "$PY" -m src.benchmarks.mme --method baseline_greedy --model "$MODEL" \
  --out-dir results/mme_baseline_greedy_internvl3_8b

run_stage mme_sbc results/mme_sbc_internvl3_8b/summary_ours_sbc.json \
  logs/internvl_mme_sbc.log \
  "$PY" -m src.benchmarks.mme --method ours_sbc --model "$MODEL" \
  --out-dir results/mme_sbc_internvl3_8b "${SBC_ARGS[@]}"

# ---------------- POPE (3 x 3000 q) ----------------
# The runner rewrites summary_<method>.json per invocation, so each split's
# summary is copied to summary_<method>_<split>.json (the resume marker) and
# the three are merged into summary_<method>.json at the end.
pope_split() {  # method dir split extra...
  local method="$1" dir="$2" split="$3"; shift 3
  "$PY" -m src.benchmarks.pope --method "$method" --setting "$split" \
    --model "$MODEL" --out-dir "$dir" "$@" \
  && cp "$dir/summary_${method}.json" "$dir/summary_${method}_${split}.json"
}
for split in random popular adversarial; do
  run_stage "pope_baseline_${split}" \
    "results/pope_baseline_greedy_internvl3_8b/summary_baseline_greedy_${split}.json" \
    "logs/internvl_pope_baseline_${split}.log" \
    pope_split baseline_greedy results/pope_baseline_greedy_internvl3_8b "$split"
done
for split in random popular adversarial; do
  run_stage "pope_sbc_${split}" \
    "results/pope_sbc_internvl3_8b/summary_ours_sbc_${split}.json" \
    "logs/internvl_pope_sbc_${split}.log" \
    pope_split ours_sbc results/pope_sbc_internvl3_8b "$split" "${SBC_ARGS[@]}"
done
"$PY" scripts/internvl_merge_summaries.py --pope-only >> "$QLOG" 2>&1 \
  && status MERGED "POPE per-split summaries -> summary_<method>.json"

# ---------------- CHAIR (1000 images, 512 new tokens) ----------------
run_stage chair_baseline results/chair_baseline_greedy_internvl3_8b/summary_baseline.json \
  logs/internvl_chair_baseline.log \
  "$PY" -m src.benchmarks.chair --method baseline --n-images 1000 --model "$MODEL" \
  --out-dir results/chair_baseline_greedy_internvl3_8b

run_stage chair_sbc results/chair_sbc_internvl3_8b/summary_ours_sbc.json \
  logs/internvl_chair_sbc.log \
  "$PY" -m src.benchmarks.chair --method ours_sbc --n-images 1000 --model "$MODEL" \
  --out-dir results/chair_sbc_internvl3_8b "${SBC_ARGS[@]}"

# ---------------- summaries ----------------
"$PY" scripts/internvl_merge_summaries.py >> "$QLOG" 2>&1 \
  && status SUMMARY "logs/internvl_decoding_summary.md"
status QUEUE_END "pid=$$"
