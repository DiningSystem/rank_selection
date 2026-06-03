#!/bin/bash

# BoolQ-focused commonsense MoE-LoRA preset without data replay/oversampling.
# This keeps the original Commonsense170K data distribution and targets BoolQ
# gains through sequence length, calibration-friendly optimization, and router
# stability rather than changing BoolQ exposure:
# - 512-token contexts preserve longer BoolQ passages.
# - Lower LR, longer warmup, and modest dropout reduce binary-label
#   overconfidence.
# - Softer router temperatures and stronger load balancing improve calibration
#   without replaying true/false examples.
# - Keep batch/accumulation large for stable gradients; if memory-constrained,
#   reduce --moe_router_hidden_dim before reducing --max_seq_length.

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train_cr.py \
  --peft_method=moe_lora \
  --epochs=2 \
  --max_seq_length=512 \
  --moe_r_max=32 \
  --moe_top_k=4 \
  --moe_router_hidden_dim=512 \
  --moe_router_norm_type=rmsnorm \
  --moe_router_activation=silu \
  --lr=7e-4 \
  --moe_router_lr=8e-5 \
  --moe_entropy_loss_weight=0.0 \
  --moe_load_balance_loss_weight=2e-4 \
  --moe_mask_l1_loss_weight=0.0 \
  --moe_router_temperature_start=2.0 \
  --moe_router_temperature_end=1.0 \
  --moe_topk_warmup_ratio=0.03 \
  --moe_aux_loss_cap=0.012 \
  --moe_aux_warmup_ratio=0.10 \
  --moe_aux_stop_ratio=0.85 \
  --moe_lora_weight_decay=0.001 \
  --moe_router_weight_decay=0.001 \
  --moe_mask_init_value=0.8 \
  --lora_dropout=0.02 \
  --warmup_ratio=0.02 \
  --scheduler=cosine \
  --adam_beta1=0.9 \
  --adam_beta2=0.98 \
  --max_grad_norm=0.2 \
  --batch_size=4 \
  --grad_acc_steps=32 \
  --lora_alpha=32 \
  --seed=100 \
  --gradient_checkpointing \
  --dataloader_num_workers=8 \
  --hf_fast_download \
  --hf_preload \
  --hf_prefer_safetensors
