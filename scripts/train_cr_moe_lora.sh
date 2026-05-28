#!/bin/bash

# Strong commonsense MoE-LoRA preset.
# Compared with the conservative stability preset, this favors accuracy:
# - 512-token contexts preserve more BoolQ passages.
# - top_k=4 gives each token more rank capacity at fixed r_max.
# - cosine decay + shorter warmup reduce underfitting from long warmup.
# If your GPU is memory-constrained, drop --max_seq_length back to 256 first.

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train_cr.py \
  --peft_method=moe_lora \
  --epochs=3 \
  --max_seq_length=512 \
  --moe_r_max=32 \
  --moe_top_k=4 \
  --moe_router_hidden_dim=512 \
  --moe_router_norm_type=rmsnorm \
  --moe_router_activation=silu \
  --lr=8e-4 \
  --moe_router_lr=1.5e-4 \
  --moe_entropy_loss_weight=0.0 \
  --moe_load_balance_loss_weight=1e-4 \
  --moe_mask_l1_loss_weight=2e-8 \
  --moe_router_temperature_start=2.0 \
  --moe_router_temperature_end=0.8 \
  --moe_topk_warmup_ratio=0.10 \
  --moe_aux_loss_cap=0.02 \
  --moe_aux_warmup_ratio=0.10 \
  --moe_aux_stop_ratio=0.65 \
  --moe_lora_weight_decay=0.002 \
  --moe_router_weight_decay=0.002 \
  --moe_mask_init_value=0.7 \
  --lora_dropout=0.03 \
  --warmup_ratio=0.08 \
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
