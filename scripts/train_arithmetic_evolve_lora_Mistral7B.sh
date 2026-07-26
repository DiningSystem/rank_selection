#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train_arithmetic.py \
  --adapter_type evolve_lora \
  --hf_fast_download --hf_preload --hf_prefer_safetensors "$@" \
  --model "mistralai/Mistral-7B-v0.1" \
  --evolve_router_hidden_dim 128 \
  --evolve_rank_delay_ratio 0 \
  --evolve_r_min 2 \
  --evolve_beta 0 \
  --evolve_alpha_max 0.0001 \
  --evolve_anneal_k 0.0 \
  --evolve_gate_floor 0.12 \
  --evolve_complexity_ema 0.9 \
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
