#!/bin/bash

# Evaluate MoE-LoRA runs that saved full Hugging Face model artifacts
# (e.g., config.json, generation_config.json, model*.safetensors).
# No adapter merge step is needed.

GPU_ID=0
BASE_MODEL=${BASE_MODEL:-""}
PREPARE_MOE_EVAL=${PREPARE_MOE_EVAL:-auto}
VLLM_BATCH_SIZE_GSM8K=${VLLM_BATCH_SIZE_GSM8K:-128}
VLLM_BATCH_SIZE_MATH=${VLLM_BATCH_SIZE_MATH:-64}
HF_BATCH_SIZE_GSM8K=${HF_BATCH_SIZE_GSM8K:-128}
HF_BATCH_SIZE_MATH=${HF_BATCH_SIZE_MATH:-32}
RUN_DIRS=()

source scripts/moe_eval_common.sh

if [ "$#" -gt 0 ]; then
  RUN_DIRS=("$@")
fi

if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
  echo "Error: No run directories provided."
  echo "Usage: bash scripts/arithmetic_eval_moe_lora.sh /abs/path/to/run_dir [/abs/path/to/run_dir2 ...]"
  exit 1
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

  BACKEND_MODE="auto"
  EFFECTIVE_GSM8K_BS="$VLLM_BATCH_SIZE_GSM8K"
  EFFECTIVE_MATH_BS="$VLLM_BATCH_SIZE_MATH"
  IS_ADAPTIVE=$(is_adaptive_checkpoint "$EVAL_MODEL_PATH")
  if [ "$IS_ADAPTIVE" = "1" ]; then
    BACKEND_MODE="hf_moe"
    EFFECTIVE_GSM8K_BS="$HF_BATCH_SIZE_GSM8K"
    EFFECTIVE_MATH_BS="$HF_BATCH_SIZE_MATH"
    echo "=== Adaptive MoE detected: using backend=$BACKEND_MODE with reduced batch sizes (${EFFECTIVE_GSM8K_BS}/${EFFECTIVE_MATH_BS}) ==="
  fi

  CUDA_VISIBLE_DEVICES=$GPU_ID python instruction_tuning_eval/MATH_eval.py \
    --model "$EVAL_MODEL_PATH" \
    --backend "$BACKEND_MODE" \
    --tokenizer "$TOKENIZER_PATH" \
    --data_file "data/math_eval/MATH_test.jsonl" \
    --batch_size "$EFFECTIVE_MATH_BS" \
    --tensor_parallel_size 1 \
    --run_dir "$RUN_ROOT"
    
  CUDA_VISIBLE_DEVICES=$GPU_ID python instruction_tuning_eval/gsm8k_eval.py \
    --model "$EVAL_MODEL_PATH" \
    --backend "$BACKEND_MODE" \
    --tokenizer "$TOKENIZER_PATH" \
    --data_file "data/math_eval/gsm8k_test.jsonl" \
    --batch_size "$EFFECTIVE_GSM8K_BS" \
    --tensor_parallel_size 1 \
    --run_dir "$RUN_ROOT"
    


done

echo "=== Done evaluating all MoE-LoRA run directories ==="
