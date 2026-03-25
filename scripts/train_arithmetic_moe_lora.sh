CUDA_VISIBLE_DEVICES=0 python train_arithmetic.py \
  --peft_method=moe_lora \
  --moe_expert_ranks=4,8,16,32 \
  --moe_top_k=2 \
  --moe_router_hidden_dim=512 \
  --lr=1e-3 \
  --hf_fast_download \
  --hf_preload \
  --hf_prefer_safetensors
