#!/bin/bash

# Evaluate MoE-LoRA runs that saved full Hugging Face model artifacts.
# No adapter merge step is needed.

GPU_ID=0
BASE_MODEL=${BASE_MODEL:-""}
PREPARE_MOE_EVAL=${PREPARE_MOE_EVAL:-auto}
VLLM_BATCH_SIZE_CR=${VLLM_BATCH_SIZE_CR:-128}
HF_BATCH_SIZE_CR=${HF_BATCH_SIZE_CR:-16}
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

  BACKEND_MODE="auto"
  EFFECTIVE_CR_BS="$VLLM_BATCH_SIZE_CR"
  IS_ADAPTIVE=$(python - "$EVAL_MODEL_PATH" <<'PY'
import json, os, sys
path = sys.argv[1]
index_path = os.path.join(path, "model.safetensors.index.json")
keys = []
if os.path.exists(index_path):
    with open(index_path, "r") as f:
        keys = list(json.load(f).get("weight_map", {}).keys())
print("1" if (any(k.endswith(".A") or k.endswith(".B") for k in keys) and any(".base.weight" in k for k in keys)) else "0")
PY
  )
  if [ "$IS_ADAPTIVE" = "1" ]; then
    BACKEND_MODE="hf_moe"
    EFFECTIVE_CR_BS="$HF_BATCH_SIZE_CR"
    echo "=== Adaptive MoE detected: using backend=$BACKEND_MODE with reduced batch size (${EFFECTIVE_CR_BS}) ==="
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
      --tensor_parallel_size 1 \
      --run_dir "$RUN_ROOT"
  done
done

echo "=== Done evaluating all MoE-LoRA run directories ==="
