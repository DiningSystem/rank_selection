#!/bin/bash

# Evaluate MoE-LoRA runs that saved full Hugging Face model artifacts
# (e.g., config.json, generation_config.json, model*.safetensors).
# No adapter merge step is needed.

GPU_ID=0
BASE_MODEL=${BASE_MODEL:-""}
PREPARE_MOE_EVAL=${PREPARE_MOE_EVAL:-auto}
RUN_DIRS=()

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

  NEED_PREPARE=0
  if [ "$PREPARE_MOE_EVAL" = "always" ]; then
    NEED_PREPARE=1
  elif [ "$PREPARE_MOE_EVAL" = "never" ]; then
    NEED_PREPARE=0
  else
    python - "$MODEL_PATH" <<'PY'
import json
import os
import sys
from safetensors.torch import load_file

model_path = sys.argv[1]
index_path = os.path.join(model_path, "model.safetensors.index.json")
single_path = os.path.join(model_path, "model.safetensors")
keys = set()
if os.path.exists(index_path):
    with open(index_path, "r") as f:
        keys = set(json.load(f).get("weight_map", {}).keys())
elif os.path.exists(single_path):
    keys = set(load_file(single_path).keys())

has_lm_head = "lm_head.weight" in keys
has_embed = ("model.embed_tokens.weight" in keys) or ("embed_tokens.weight" in keys)
has_norm = ("model.norm.weight" in keys) or ("norm.weight" in keys)
sys.exit(0 if (has_lm_head and has_embed and has_norm) else 1)
PY
    if [ $? -ne 0 ]; then
      NEED_PREPARE=1
    fi
  fi

  if [ "$NEED_PREPARE" -eq 1 ]; then
    echo "=== Preparing eval-ready MoE full model (base + MoE checkpoint) ==="
    PREPARE_ARGS=(--checkpoint_dir "$MODEL_PATH" --output_dir "$EVAL_MODEL_PATH")
    if [ -n "$BASE_MODEL" ]; then
      PREPARE_ARGS+=(--base_model "$BASE_MODEL")
    fi
    python scripts/prepare_moe_eval_model.py "${PREPARE_ARGS[@]}"
  else
    echo "=== Checkpoint appears full; skipping base+MoE preparation ==="
    EVAL_MODEL_PATH="$MODEL_PATH"
  fi

  CUDA_VISIBLE_DEVICES=$GPU_ID python instruction_tuning_eval/gsm8k_eval.py \
    --model "$EVAL_MODEL_PATH" \
    --backend "auto" \
    --tokenizer "$TOKENIZER_PATH" \
    --data_file "data/math_eval/gsm8k_test.jsonl" \
    --batch_size 128 \
    --tensor_parallel_size 1 \
    --run_dir "$RUN_ROOT"

  CUDA_VISIBLE_DEVICES=$GPU_ID python instruction_tuning_eval/MATH_eval.py \
    --model "$EVAL_MODEL_PATH" \
    --backend "auto" \
    --tokenizer "$TOKENIZER_PATH" \
    --data_file "data/math_eval/MATH_test.jsonl" \
    --batch_size 64 \
    --tensor_parallel_size 1 \
    --run_dir "$RUN_ROOT"
done

echo "=== Done evaluating all MoE-LoRA run directories ==="
