#!/bin/bash

# Strong commonsense MoE-LoRA preset.
# This version keeps the original Commonsense170K data distribution and improves
# BoolQ/ARC-Challenge through capacity and optimization parameters only:
# - 512-token contexts preserve longer BoolQ passages.
# - r_max=32 and lora_alpha=32 preserve the original adapter budget.
# - top_k=4, a larger router MLP, and lower dropout use that fixed budget better
#   for BoolQ/ARC-Challenge without changing dataset exposure.
# - cosine decay, shorter warmup, and a cooler final router sharpen the two-epoch
#   run while keeping routing stable.
# If your GPU is memory-constrained, reduce --moe_router_hidden_dim to 384 before
# reducing --max_seq_length, because BoolQ is sensitive to context truncation.

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train_cr.py \
  --peft_method=moe_lora \
  --epochs=2 \
  --max_seq_length=512 \
  --moe_r_max=32 \
  --moe_top_k=4 \
  --moe_router_hidden_dim=512 \
  --moe_router_norm_type=rmsnorm \
  --moe_router_activation=silu \
  --lr=8e-4 \
  --moe_router_lr=1.2e-4 \
  --moe_entropy_loss_weight=0.0 \
  --moe_load_balance_loss_weight=1e-4 \
  --moe_mask_l1_loss_weight=1e-8 \
  --moe_router_temperature_start=1.8 \
  --moe_router_temperature_end=0.75 \
  --moe_topk_warmup_ratio=0.08 \
  --moe_aux_loss_cap=0.015 \
  --moe_aux_warmup_ratio=0.08 \
  --moe_aux_stop_ratio=0.60 \
  --moe_lora_weight_decay=0.001 \
  --moe_router_weight_decay=0.001 \
  --moe_mask_init_value=0.8 \
  --lora_dropout=0.02 \
  --warmup_ratio=0.05 \
  --scheduler=cosine \
  --adam_beta1=0.9 \
  --adam_beta2=0.98 \
  --max_grad_norm=0.2 \
  --batch_size=4 \
  --grad_acc_steps=32 \
  --lora_alpha=32 \
  --seed=123 \
  --gradient_checkpointing \
  --dataloader_num_workers=16 \
  --hf_fast_download \
  --hf_preload \
  --hf_prefer_safetensors
