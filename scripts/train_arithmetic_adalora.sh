#!/bin/bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train_arithmetic.py \
  --adapter_type adalora \
  --lora_r 8 \
  --adalora_target_r 4 \
  --lora_alpha 8 \
  --lora_dropout 0 \
  --lr 8e-4 \
  --batch_size 8 \
  --hf_fast_download --hf_preload --hf_prefer_safetensors \
  "$@"
