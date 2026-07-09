#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-"google/gemma-2-9b"}
GPU_ID=${GPU_ID:-0}
BATCH_SIZE=${BATCH_SIZE:-1}
TORCH_DTYPE=${TORCH_DTYPE:-"bfloat16"}
DEVICE_MAP=${DEVICE_MAP:-"auto"}
GSM8K_MAX_NEW_TOKENS=${GSM8K_MAX_NEW_TOKENS:-256}
MATH_MAX_NEW_TOKENS=${MATH_MAX_NEW_TOKENS:-512}

if [[ $# -lt 1 ]]; then
  cat <<USAGE >&2
Usage:
  bash $0 RUN_DIR [RUN_DIR2 ...]

Each RUN_DIR may be either a training run directory containing final_model/ or the final_model directory itself.
Evaluates GSM8K and MATH using Hugging Face adapter inference without merging input-conditioned evolve-LoRA weights.
USAGE
  exit 2
fi

for RAW_RUN_DIR in "$@"; do
  RUN_DIR="$(printf '%s' "$RAW_RUN_DIR" | sed 's/\r$//')"
  RUN_DIR="${RUN_DIR%/}"

  if [[ "$(basename "$RUN_DIR")" == "final_model" ]]; then
    FINAL_MODEL_PATH="$RUN_DIR"
    LOG_RUN_DIR="$(dirname "$RUN_DIR")"
  else
    FINAL_MODEL_PATH="$RUN_DIR/final_model"
    LOG_RUN_DIR="$RUN_DIR"
  fi

  echo "=== Processing evolve-LoRA arithmetic adapter: $FINAL_MODEL_PATH ==="
  if [[ ! -d "$FINAL_MODEL_PATH" ]]; then
    echo "Error: final_model directory not found at $FINAL_MODEL_PATH" >&2
    continue
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" python instruction_tuning_eval/arithmetic_eval_hf_evolve_lora.py \
    --base_model "$MODEL" \
    --adapter_path "$FINAL_MODEL_PATH" \
    --task gsm8k \
    --data_file "data/math_eval/gsm8k_test.jsonl" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$GSM8K_MAX_NEW_TOKENS" \
    --torch_dtype "$TORCH_DTYPE" \
    --device_map "$DEVICE_MAP" \
    --run_dir "$LOG_RUN_DIR"

  CUDA_VISIBLE_DEVICES="$GPU_ID" python instruction_tuning_eval/arithmetic_eval_hf_evolve_lora.py \
    --base_model "$MODEL" \
    --adapter_path "$FINAL_MODEL_PATH" \
    --task math \
    --data_file "data/math_eval/MATH_test.jsonl" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MATH_MAX_NEW_TOKENS" \
    --torch_dtype "$TORCH_DTYPE" \
    --device_map "$DEVICE_MAP" \
    --run_dir "$LOG_RUN_DIR"

  cat <<EOF > "$LOG_RUN_DIR/evolve_lora_hf_arithmetic_eval_info.txt"
Run processed at: $(date)
Base model: $MODEL
Backend: Hugging Face adapter inference
Datasets: gsm8k math
EOF
  echo "=== Completed evolve-LoRA HF arithmetic eval for $LOG_RUN_DIR ==="
done
