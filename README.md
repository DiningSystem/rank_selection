# ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models

Code for the paper [ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](https://arxiv.org/abs/2505.14238)

## Introduction
Large Language Models have demonstrated strong performance across a wide range of tasks, but adapting them efficiently to new domains remains a key challenge. Parameter-Efficient Fine-Tuning (PEFT) methods address this by introducing lightweight, trainable modules while keeping most pre-trained weights fixed. The prevailing approach, LoRA, models updates using a low-rank decomposition, but its expressivity is inherently constrained by the rank. Recent methods like HiRA aim to increase expressivity by incorporating a Hadamard product with the frozen weights, but still rely on the structure of the pre-trained model.
We introduce **ABBA**, a new PEFT architecture that reparameterizes the update as a Hadamard product of two independently learnable low-rank matrices. In contrast to prior work, ABBA fully decouples the update from the pre-trained weights, enabling both components to be optimized freely. This leads to significantly higher expressivity under the same parameter budget. We formally analyze ABBA’s expressive capacity and validate its advantages through matrix reconstruction experiments. 
Empirically, ABBA achieves state-of-the-art results on arithmetic and commonsense reasoning benchmarks, consistently outperforming existing PEFT methods by a significant margin across multiple models.

![intro-fig](assets/abba_github.jpg)


## Environment

We recommend using a Conda environment to run the Python scripts for this project. Follow these commands to set up the environment and install the required libraries:

```bash
conda create -n abba python=3.10
conda activate abba
pip install -r requirements.txt
```

If model download from Hugging Face is slow (for example `mistralai/Mistral-7B-v0.1`), install the optional transfer backend and use the new CLI flags:

```bash
pip install hf_transfer
bash scripts/train_arithmetic.sh
```

You can also set a persistent cache location and reuse local files:

```bash
python train_arithmetic.py \
  --model mistralai/Mistral-7B-v0.1 \
  --hf_fast_download \
  --hf_preload \
  --hf_prefer_safetensors \
  --hf_download_workers 16 \
  --hf_cache_dir /path/to/hf-cache \
  --hf_local_files_only
```

`--hf_prefer_safetensors` attempts to avoid slow `.bin` downloads by preferring safetensors files (with automatic fallback if a model repo does not provide safetensors).

## Arithmetic Reasoning

To train the models, execute:

```bash
bash scripts/train_arithmetic.sh
```

This script will fine-tune a model on the MetaMathQA dataset. You can modify the `model` parameter to use a different model if desired. The script will save the fine-tuned adapters.

Run the following to evaluate on GSM8K and MATH benchmarks:
```bash
bash scripts/arithmetic_merge_eval.sh /abs/path/to/run_dir
```

`scripts/arithmetic_merge_eval.sh` now accepts one or more run directories as CLI arguments. You can pass either:

- a run directory (for example `.../20260324_060905_rank_32_lr0.001_alpha_32_train_train50000`), or
- a direct `.../final_model` directory path.

Examples:

```bash
# Single run directory
bash scripts/arithmetic_merge_eval.sh \
  /home/gdi-user/enguyen/research_vllm/test/rank_selection/experiments/arithmetic/Mistral-7B-v0.1/20260324_060905_rank_32_lr0.001_alpha_32_train_train50000

# Direct final_model path
bash scripts/arithmetic_merge_eval.sh \
  /home/gdi-user/enguyen/research_vllm/test/rank_selection/experiments/arithmetic/Mistral-7B-v0.1/20260324_060905_rank_32_lr0.001_alpha_32_train_train50000/final_model

# Multiple runs
bash scripts/arithmetic_merge_eval.sh /abs/path/to/run1 /abs/path/to/run2
```

## Commonsense Reasoning

To run the commonsense experiments, start by downloading the required datasets.

Begin by fetching the fine-tuning dataset available [here](XXXX). Place this file in the `data/commonsense` folder.

Next, for the evaluation phase, download the necessary datasets from [this link](XXXX). Ensure each dataset is saved in its appropriate subdirectory within `data/commonsense`.

To train the models, use:

```bash
bash scripts/train_cr.sh
```

This script will fine-tune a model on the Commonsense170K dataset. You can modify the `model` parameter to explore various models. The script will save the fine-tuned adapters.

For MoE-LoRA CR training, the accuracy-oriented preset is:

```bash
bash scripts/train_cr_moe_lora.sh
```

This preset is tuned for the common failure mode where BoolQ and ARC-Challenge lag the other commonsense datasets while keeping training capped at `--epochs=2`, preserving the original Commonsense170K data distribution, and retaining the original adapter budget with `--moe_r_max=32` and `--lora_alpha=32`. It improves through the other parameters: `--max_seq_length=512` preserves longer BoolQ passages, `--moe_top_k=4` plus a `512`-hidden router uses the fixed rank budget more selectively, and the cosine schedule, shorter warmup, lower dropout, and cooler final router temperature make the two-epoch run sharper. The router defaults remain LLaMA-friendly:

- `--moe_router_norm_type=rmsnorm`
- `--moe_router_activation=silu`

If the run is unstable or your GPU runs out of memory, try these in order: reduce `--moe_router_hidden_dim` to `384`, reduce `--moe_top_k` to `3`, or reduce `--max_seq_length` back to `256`. If BoolQ is still low, keep `--max_seq_length=512` and try `--lr=6e-4`; if ARC-Challenge is still low, keep `--moe_r_max=32`/`--lora_alpha=32` and try `--moe_top_k=5` with `--moe_router_lr=1e-4`.

Run the following to evaluate on commonsense reasoning benchmarks:
```bash
bash scripts/cr_merge_eval.sh
```

## Citation

If you use our work, please cite us:

```
@article{singhal2025abba,
  title={ABBA: Highly Expressive Hadamard Product Adaptation for Large Language Models},
  author={Singhal, Raghav and Ponkshe, Kaustubh and Vartak, Rohit and Vepakomma, Praneeth},
  journal={arXiv preprint arXiv:2505.14238},
  year={2025}
}

@article{ponkshe2024initialization,
  title={Initialization using update approximation is a silver bullet for extremely efficient low-rank fine-tuning},
  author={Ponkshe, Kaustubh and Singhal, Raghav and Gorbunov, Eduard and Tumanov, Alexey and Horvath, Samuel and Vepakomma, Praneeth},
  journal={arXiv preprint arXiv:2411.19557},
  year={2024}
}
```
