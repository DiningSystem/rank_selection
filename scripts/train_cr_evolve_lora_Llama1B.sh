CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train_cr.py \
  --adapter_type evolve_lora \
  --lora_r 32 \
  --hf_fast_download --hf_preload --hf_prefer_safetensors "$@" \
  --model "meta-llama/Llama-3.2-3B" \
  --lr 5.0e-4
