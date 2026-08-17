#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train_arithmetic.py \
  --adapter_type evolve_lora \
  --hf_fast_download --hf_preload --hf_prefer_safetensors "$@" \
  --model "google/gemma-2-9b" \
  --evolve_rank_delay_ratio 0 \
  --evolve_r_min 2 \
  --evolve_beta 0 \
  --evolve_alpha_max 0.00007 \
  --evolve_anneal_k 0 \
  --evolve_gate_floor 0 \
  --ortho_weight 7e-5 \
  --evolve_active_component_threshold 0.01 \
  --evolve_active_log_max_layers 6 \
  --lora_r 32 \
  --lora_alpha 32 \
  --lora_dropout 0 \
  --batch_size 1 \
  --grad_acc_steps 32 \
  --epochs 1 \
  --scheduler cosine \
  --warmup_ratio 0.02 \
  --max_seq_length 512 \
  --lr 1e-3
