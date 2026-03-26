#!/bin/bash

# Evaluate MoE-LoRA runs that saved full Hugging Face model artifacts.
# No adapter merge step is needed.

GPU_ID=0
RUN_DIRS=()

if [ "$#" -gt 0 ]; then
  RUN_DIRS=("$@")
fi

if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
  echo "Error: No run directories provided."
  echo "Usage: bash scripts/cr_eval_moe_lora.sh /abs/path/to/run_dir [/abs/path/to/run_dir2 ...]"
  exit 1
fi

declare -a DATASETS=(
  "ARC-Challenge"
  "ARC-Easy"
  "boolq"
  "hellaswag"
  "openbookqa"
  "piqa"
  "social_i_qa"
  "winogrande"
)

for RAW_RUN_DIR in "${RUN_DIRS[@]}"; do
  RUN_DIR="$(printf '%s' "$RAW_RUN_DIR" | sed 's/\r$//')"
  RUN_DIR="${RUN_DIR%/}"

  if [ "$(basename "$RUN_DIR")" = "final_model" ]; then
    MODEL_PATH="$RUN_DIR"
    RUN_ROOT="$(dirname "$RUN_DIR")"
  else
    MODEL_PATH="$RUN_DIR/final_model"
    RUN_ROOT="$RUN_DIR"
  fi
  TOKENIZER_PATH="$MODEL_PATH"
  if [ -d "$RUN_ROOT/tokenizer" ]; then
    TOKENIZER_PATH="$RUN_ROOT/tokenizer"
  fi
  MODEL_FOR_EVAL="$MODEL_PATH"

  INDEX_JSON="$MODEL_PATH/model.safetensors.index.json"
  if [ -f "$INDEX_JSON" ] && grep -q "\"model.layers\\..*\\.A\"" "$INDEX_JSON"; then
    FIXED_MODEL_PATH="$RUN_ROOT/final_model_prefix_fixed"
    if [ ! -d "$FIXED_MODEL_PATH" ]; then
      echo "Detected model-prefixed Rank-MoE keys; creating normalized checkpoint at $FIXED_MODEL_PATH"
      python scripts/normalize_moe_safetensors_prefix.py \
        --input_dir "$MODEL_PATH" \
        --output_dir "$FIXED_MODEL_PATH"
    fi
    MODEL_FOR_EVAL="$FIXED_MODEL_PATH"
  fi

  echo "=== Evaluating MoE-LoRA run: $RUN_ROOT ==="
  if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: final model directory not found at $MODEL_PATH"
    continue
  fi

  for dataset in "${DATASETS[@]}"; do
    echo "--- Dataset: $dataset ---"
    CUDA_VISIBLE_DEVICES=$GPU_ID python instruction_tuning_eval/commonsense_eval.py \
      --model "$MODEL_FOR_EVAL" \
      --tokenizer "$TOKENIZER_PATH" \
      --dataset "$dataset" \
      --data_file "data/commonsense/$dataset/test.json" \
      --batch_size 128 \
      --tensor_parallel_size 1 \
      --run_dir "$RUN_ROOT"
  done
done

echo "=== Done evaluating all MoE-LoRA run directories ==="
