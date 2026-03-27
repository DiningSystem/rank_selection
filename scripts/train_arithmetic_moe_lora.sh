CUDA_VISIBLE_DEVICES=0 python train_arithmetic.py \
  --peft_method=moe_lora \
  --moe_r_max=32 \
  --moe_top_k=1 \
  --moe_router_hidden_dim=128 \
  --lora_alpha=32\
  --lr=1e-3 \
  --hf_fast_download \
  --hf_preload \
  --hf_prefer_safetensors
