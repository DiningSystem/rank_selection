#!/bin/bash

# BoolQ retrain preset without data replay/oversampling.
# Use this after the previous no-replay presets plateau around 70-71. This
# version is allowed to train for longer, but it keeps the number of optimizer
# updates matched to the baseline default: baseline steps are proportional to
#   2 epochs / (batch_size 6 * grad_acc_steps 24).
# With 3 epochs we therefore set batch_size * grad_acc_steps = 216. Use
# batch_size=6 and grad_acc_steps=36 so the per-device batch matches the
# baseline default while the reduced accumulation keeps the optimizer-step
# count unchanged. Do not increase the effective batch above 216 just to chase
# accuracy: that lowers optimizer steps and has tended to hurt BoolQ recovery.
# If memory is tight at 768 tokens, fall back to batch_size=3/grad_acc_steps=72
# or batch_size=2/grad_acc_steps=108, keeping the product fixed.
# The longer pass count gives BoolQ more exposure without replay, while the
# matched effective batch preserves the baseline optimizer-step budget.
# Config intent:
# - 768-token contexts keep useful BoolQ passage context without the instability
#   seen from the 1024/top_k=6 capacity-heavy run.
# - top_k=3 and a 512-hidden router are a middle ground between under-routing
#   (top_k=2) and over-routing binary examples (top_k=6).
# - Moderate LR, longer warmup, dropout, and soft final router temperature target
#   true/false calibration rather than memorization.

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train_cr.py \
  --peft_method=moe_lora \
  --epochs=3 \
  --max_seq_length=768 \
  --moe_r_max=32 \
  --moe_top_k=3 \
  --moe_router_hidden_dim=512 \
  --moe_router_norm_type=rmsnorm \
  --moe_router_activation=silu \
  --lr=7e-4 \
  --moe_router_lr=5e-5 \
  --moe_entropy_loss_weight=0.0 \
  --moe_load_balance_loss_weight=3e-4 \
  --moe_mask_l1_loss_weight=0.0 \
  --moe_router_temperature_start=2.5 \
  --moe_router_temperature_end=1.15 \
  --moe_topk_warmup_ratio=0.08 \
  --moe_aux_loss_cap=0.012 \
  --moe_aux_warmup_ratio=0.12 \
  --moe_aux_stop_ratio=0.90 \
  --moe_lora_weight_decay=0.0015 \
  --moe_router_weight_decay=0.0015 \
  --moe_mask_init_value=0.8 \
  --lora_dropout=0.03 \
  --warmup_ratio=0.04 \
  --scheduler=cosine \
  --adam_beta1=0.9 \
  --adam_beta2=0.98 \
  --max_grad_norm=0.2 \
  --batch_size=6 \
  --grad_acc_steps=36 \
  --lora_alpha=32 \
  --seed=100 \
  --gradient_checkpointing \
  --dataloader_num_workers=16 \
  --hf_fast_download \
  --hf_preload \
  --hf_prefer_safetensors
