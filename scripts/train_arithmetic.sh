CUDA_VISIBLE_DEVICES=0 python train_arithmetic.py --model="google/gemma-2-9b" --lora_r=32 --lr=1e-3 --lora_alpha=32 --hf_fast_download --hf_preload --hf_prefer_safetensors

