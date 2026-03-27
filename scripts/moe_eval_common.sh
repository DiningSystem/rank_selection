#!/bin/bash

# Shared helpers for MoE evaluation shell scripts.

is_full_checkpoint() {
  local model_path="$1"
  python - "$model_path" <<'PY'
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
}

is_adaptive_checkpoint() {
  local model_path="$1"
  python - "$model_path" <<'PY'
import json
import os
import sys

path = sys.argv[1]
index_path = os.path.join(path, "model.safetensors.index.json")
keys = []
if os.path.exists(index_path):
    with open(index_path, "r") as f:
        keys = list(json.load(f).get("weight_map", {}).keys())
print("1" if (any(k.endswith(".A") or k.endswith(".B") for k in keys) and any(".base.weight" in k for k in keys)) else "0")
PY
}

prepare_eval_model_if_needed() {
  local model_path="$1"
  local run_root="$2"
  local base_model="$3"
  local prepare_mode="$4"
  local eval_model_path="$5"

  local need_prepare=0
  if [ "$prepare_mode" = "always" ]; then
    need_prepare=1
  elif [ "$prepare_mode" = "never" ]; then
    need_prepare=0
  else
    if ! is_full_checkpoint "$model_path"; then
      need_prepare=1
    fi
  fi

  if [ "$need_prepare" -eq 1 ]; then
    echo "=== Preparing eval-ready MoE full model (base + MoE checkpoint) ==="
    PREPARE_ARGS=(--checkpoint_dir "$model_path" --output_dir "$eval_model_path")
    if [ -n "$base_model" ]; then
      PREPARE_ARGS+=(--base_model "$base_model")
    fi
    python scripts/prepare_moe_eval_model.py "${PREPARE_ARGS[@]}"
    echo "$eval_model_path"
  else
    echo "=== Checkpoint appears full; skipping base+MoE preparation ==="
    echo "$model_path"
  fi
}
