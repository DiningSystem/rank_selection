#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-"meta-llama/Llama-3.2-3B"}
GPU_ID=${GPU_ID:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
TORCH_DTYPE=${TORCH_DTYPE:-"bfloat16"}
DEVICE_MAP=${DEVICE_MAP:-"auto"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}

if [[ $# -lt 1 ]]; then
  cat <<'USAGE' >&2
Usage:
  MODEL=meta-llama/Llama-3.2-3B bash scripts/cr_evolve_lora_hf_eval.sh RUN_DIR [RUN_DIR2 ...]

Each RUN_DIR may be either a training run directory containing final_model/ or the final_model directory itself.
Evaluates the same commonsense datasets as scripts/cr_merge_eval.sh using HF adapter inference.
USAGE
  exit 2
fi

DATASETS=(
  "ARC-Challenge"
  "ARC-Easy"
  "boolq"
  "hellaswag"
  "openbookqa"
  "piqa"
  "social_i_qa"
  "winogrande"
)

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

  echo "=== Processing evolve-LoRA adapter: $FINAL_MODEL_PATH ==="
  if [[ ! -d "$FINAL_MODEL_PATH" ]]; then
    echo "Error: final_model directory not found at $FINAL_MODEL_PATH" >&2
    continue
  fi

  for dataset in "${DATASETS[@]}"; do
    echo "=== Evaluating evolve-LoRA on $dataset with HF backend ==="
    CUDA_VISIBLE_DEVICES="$GPU_ID" python instruction_tuning_eval/commonsense_eval_hf_evolve_lora.py \
      --base_model "$MODEL" \
      --adapter_path "$FINAL_MODEL_PATH" \
      --dataset "$dataset" \
      --data_file "data/commonsense/$dataset/test.json" \
      --batch_size "$BATCH_SIZE" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --torch_dtype "$TORCH_DTYPE" \
      --device_map "$DEVICE_MAP" \
      --run_dir "$LOG_RUN_DIR"
  done

  cat <<EOF > "$LOG_RUN_DIR/evolve_lora_hf_eval_info.txt"
Run processed at: $(date)
Base model: $MODEL
Backend: Hugging Face adapter inference
Datasets: ${DATASETS[*]}
EOF
  echo "=== Completed evolve-LoRA HF commonsense eval for $LOG_RUN_DIR ==="
done
