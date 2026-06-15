#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<'USAGE' >&2
Usage:
  bash scripts/infer_evolve_lora_hf.sh BASE_MODEL ADAPTER_PATH PROMPT [extra inference args]

Example:
  bash scripts/infer_evolve_lora_hf.sh \
    mistralai/Mistral-7B-v0.1 \
    experiments/arithmetic/Mistral-7B-v0.1/<run>/final_model \
    "Solve: 17 * 23 =" \
    --max_new_tokens 128 --torch_dtype bfloat16
USAGE
  exit 2
fi

BASE_MODEL="$1"
ADAPTER_PATH="$2"
PROMPT="$3"
shift 3

python inference_evolve_lora.py \
  --backend hf \
  --model "$BASE_MODEL" \
  --adapter_path "$ADAPTER_PATH" \
  --prompt "$PROMPT" \
  "$@"
