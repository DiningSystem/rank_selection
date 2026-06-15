CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python train_cr.py \
  --adapter_type evolve_lora \
  --lora_r 32 \
  --lora_alpha 32 \
  --evolve_r_min 2 \
  --evolve_beta 0.05 \
  --evolve_alpha_max 0.01 \
  --evolve_anneal_k 5e-5 \
  --lr=1e-3 \
  --hf_fast_download --hf_preload --hf_prefer_safetensors "$@"
