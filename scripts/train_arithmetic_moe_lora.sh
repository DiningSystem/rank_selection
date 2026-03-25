CUDA_VISIBLE_DEVICES=0 python train_arithmetic.py \
  --peft_method=moe_lora \
  --moe_expert_ranks=4,8 \
  --moe_top_k=1 \
  --moe_router_hidden_dim=0 \
  --gradient_checkpointing \
  --lr=1e-3 \
  --hf_fast_download \
  --hf_preload \
  --hf_prefer_safetensors
