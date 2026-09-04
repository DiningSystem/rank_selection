#!/bin/bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train_cr.py \
  --adapter_type sora \
  --lora_r 8 \
  --lora_alpha 8 \
  --lora_dropout 0 \
  --lr 8e-4 \
  --sora_gate_lr 0.1 \
  --sora_lambda_sparsity 0.1 \
  --batch_size 8 \
  --hf_fast_download --hf_preload --hf_prefer_safetensors \
  "$@"
