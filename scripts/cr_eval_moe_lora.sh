#!/bin/bash

# Evaluate MoE-LoRA runs that saved full Hugging Face model artifacts.
# No adapter merge step is needed.

GPU_ID=0
BASE_MODEL=${BASE_MODEL:-""}
PREPARE_MOE_EVAL=${PREPARE_MOE_EVAL:-auto}
VLLM_BATCH_SIZE_CR=${VLLM_BATCH_SIZE_CR:-128}
# HF MoE keeps the model in eager PyTorch and is more memory-sensitive than vLLM.
# Keep the default conservative; override HF_BATCH_SIZE_CR upward on larger GPUs.
HF_BATCH_SIZE_CR=${HF_BATCH_SIZE_CR:-16}
EVAL_BACKEND=${EVAL_BACKEND:-hf_moe}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.0}
EVAL_TOP_P=${EVAL_TOP_P:-1.0}
EVAL_TOP_K=${EVAL_TOP_K:--1}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-32}
EVAL_MODE=${EVAL_MODE:-generate}
EVAL_CHOICE_TEMPLATE=${EVAL_CHOICE_TEMPLATE:-the correct answer is {choice}}
EVAL_NO_NORMALIZE_CHOICES=${EVAL_NO_NORMALIZE_CHOICES:-0}
HF_MOE_SCORE_BATCH_SIZE=${HF_MOE_SCORE_BATCH_SIZE:-16}
export HF_MOE_SCORE_BATCH_SIZE
EVAL_DATASETS=${EVAL_DATASETS:-}
EVAL_SAVE_PREDICTIONS=${EVAL_SAVE_PREDICTIONS:-0}
EVAL_PREDICTION_OUTPUT_DIR=${EVAL_PREDICTION_OUTPUT_DIR:-}
RUN_DIRS=()

source scripts/moe_eval_common.sh

if [ "$#" -gt 0 ]; then
  RUN_DIRS=("$@")
fi

if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
  echo "Error: No run directories provided."
  echo "Usage: bash scripts/cr_eval_moe_lora.sh /abs/path/to/run_dir [/abs/path/to/run_dir2 ...]"
  exit 1
fi

if [ -n "$EVAL_DATASETS" ]; then
  read -r -a DATASETS <<< "$EVAL_DATASETS"
else
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
fi

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
  EVAL_MODEL_PATH="$RUN_ROOT/final_model_eval_ready"

  echo "=== Evaluating MoE-LoRA run: $RUN_ROOT ==="
  if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: final model directory not found at $MODEL_PATH"
    continue
  fi

  EVAL_MODEL_PATH=$(prepare_eval_model_if_needed "$MODEL_PATH" "$RUN_ROOT" "$BASE_MODEL" "$PREPARE_MOE_EVAL" "$EVAL_MODEL_PATH" | tail -n 1)

  BACKEND_MODE="$EVAL_BACKEND"
  EFFECTIVE_CR_BS="$VLLM_BATCH_SIZE_CR"
  if [ "$BACKEND_MODE" = "hf_moe" ]; then
    EFFECTIVE_CR_BS="$HF_BATCH_SIZE_CR"
  fi
  echo "=== Using backend=$BACKEND_MODE for MoE eval (batch_size=${EFFECTIVE_CR_BS}) ==="
  EXTRA_EVAL_ARGS=()
  if [ "$EVAL_NO_NORMALIZE_CHOICES" = "1" ]; then
    EXTRA_EVAL_ARGS+=(--no_normalize_choices)
  fi
  if [ "$EVAL_SAVE_PREDICTIONS" = "1" ]; then
    EXTRA_EVAL_ARGS+=(--save_predictions)
    if [ -n "$EVAL_PREDICTION_OUTPUT_DIR" ]; then
      EXTRA_EVAL_ARGS+=(--prediction_output_dir "$EVAL_PREDICTION_OUTPUT_DIR")
    fi
  fi

  for dataset in "${DATASETS[@]}"; do
    echo "--- Dataset: $dataset ---"
    CUDA_VISIBLE_DEVICES=$GPU_ID python instruction_tuning_eval/commonsense_eval.py \
      --model "$EVAL_MODEL_PATH" \
      --backend "$BACKEND_MODE" \
      --tokenizer "$TOKENIZER_PATH" \
      --dataset "$dataset" \
      --data_file "data/commonsense/$dataset/test.json" \
      --batch_size "$EFFECTIVE_CR_BS" \
      --temperature "$EVAL_TEMPERATURE" \
      --top_p "$EVAL_TOP_P" \
      --top_k "$EVAL_TOP_K" \
      --max_tokens "$EVAL_MAX_TOKENS" \
      --eval_mode "$EVAL_MODE" \
      --choice_template "$EVAL_CHOICE_TEMPLATE" \
      --tensor_parallel_size 1 \
      --run_dir "$RUN_ROOT" \
      "${EXTRA_EVAL_ARGS[@]}"
  done
done

echo "=== Done evaluating all MoE-LoRA run directories ==="
