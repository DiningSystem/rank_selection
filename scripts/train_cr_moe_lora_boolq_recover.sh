#!/bin/bash

# BoolQ recovery preset for the case where all commonsense datasets are healthy
# but BoolQ stalls around ~70% with a true-label bias.
# Main changes from train_cr_moe_lora.sh:
# - Mildly lower LR and longer warmup to reduce binary-label overconfidence.
# - Softer final router temperature for better true/false calibration.
# - Conservative BoolQ replay, with extra false-label replay to counter the
#   observed prediction distribution skew toward true.
# - Slightly smaller HF input worker count for shared machines; raise if desired.

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train_cr.py \
  --peft_method=moe_lora \
  --epochs=2 \
  --max_seq_length=512 \
  --boolq_oversample_factor=2 \
  --boolq_false_oversample_factor=3 \
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
