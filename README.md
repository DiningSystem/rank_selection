# Rank Selection / Evolve-LoRA

Code for efficient fine-tuning experiments with **input-conditioned spectral LoRA (evolve-LoRA)** and the original ABBA adapter implementation.

> **Branch:** `evolve_rank`
>
> The `evolve_rank` branch contains the current evolve-LoRA implementation and training/inference utilities described below.

## Overview

The evolve-LoRA adapter parameterizes the weight update as

\[
\Delta W(x) = U\,\mathrm{diag}(\lambda(x))\,V^T,
\]

where `U` and `V` are learned low-rank factors and the spectral coefficients `lambda(x)` are predicted from the input by a lightweight router. This makes the effective adapter rank **input-dependent** rather than fixed for every example.

The current implementation uses a softmax router with a configurable gate floor:

\[
\lambda(x) = g_{\min} + (1-g_{\min})\,\mathrm{softmax}(R(x)).
\]

The adapter is therefore dynamic at inference time and cannot be exactly merged into a single static weight matrix.

![intro-fig](assets/abba_github.jpg)

## Repository structure

Important files for evolve-LoRA:

```text
evolve_lora.py                 # Evolve-LoRA module, router, rank metrics, Trainer
train_arithmetic.py             # Training entrypoint
inference_evolve_lora.py        # Hugging Face / vLLM inference entrypoint
scripts/
  train_arithmetic_evolve_lora.sh
  infer_evolve_lora_hf.sh
  arithmetic_merge_eval.sh
  cr_merge_eval.sh
```

The repository also contains the original ABBA implementation and the existing arithmetic/commonsense evaluation pipelines.

## Environment

We recommend using a Conda environment:

```bash
conda create -n abba python=3.10
conda activate abba
pip install -r requirements.txt
```

If Hugging Face model downloads are slow, install the optional transfer backend:

```bash
pip install hf_transfer
```

The training entrypoints support accelerated/preloaded Hugging Face downloads and safetensors preference. For example:

```bash
python train_arithmetic.py \\
  --model mistralai/Mistral-7B-v0.1 \\
  --adapter_type evolve_lora \\
  --hf_fast_download \\
  --hf_preload \\
  --hf_prefer_safetensors \\
  --hf_download_workers 16 \\
  --hf_cache_dir /path/to/hf-cache
```

For offline/local-cache execution, add:

```bash
--hf_local_files_only
```

## Evolve-LoRA

### Method

For a base linear layer \(W\), evolve-LoRA keeps the base weights frozen and adds

\[
y = Wx + \left(xV \odot \lambda(x)\right)U^T \cdot \frac{\alpha}{r_{\max}},
\]

where:

- `U ∈ R^{d_out × r_max}` is the output factor.
- `V ∈ R^{d_in × r_max}` is the input factor.
- `r_max` is the maximum adapter rank.
- `lambda(x)` is produced by the input-conditioned router.
- `alpha / r_max` is the adapter scaling factor.
- `gate_floor` prevents spectral coefficients from reaching zero.

The router is a lightweight linear projection from the layer input to `r_max` spectral components. Router inputs can optionally be detached from the computation graph with `--evolve_no_detach_router` controlling this behavior.

### Rank statistics

The implementation reports an entropy-based effective rank:

\[
r_{\mathrm{eff}} = \exp\left(-\sum_i p_i\log p_i\right),
\]

where the current implementation uses the spectral coefficients directly as `p_i`.

It also records the number of active spectral components above `--evolve_active_component_threshold`. These metrics are logged under the `evolve/*` namespace in Weights & Biases.

### Current loss behavior

The current `evolve_rank` implementation uses the **task loss only** for optimization:

\[
\mathcal{L} = \mathcal{L}_{task}.
\]

The rank, entropy, balance, and orthogonality utilities remain available in `evolve_lora.py` for experimentation, but the corresponding regularization terms are currently disabled in `EvolveLoRATrainer.compute_loss`. The reported effective rank and active-component statistics are monitoring metrics rather than optimization losses.

## Arithmetic reasoning

### Training

The provided arithmetic wrapper launches `train_arithmetic.py` with `--adapter_type evolve_lora`:

```bash
bash scripts/train_arithmetic_evolve_lora.sh \\
  --model mistralai/Mistral-7B-v0.1 \\
  --dataset_split 'train[:50000]'
```

The current wrapper uses:

```text
lora_r = 32
lora_alpha = 16
evolve_r_min = 2
evolve_beta = 0.05
evolve_alpha_max = 0.01
evolve_anneal_k = 5e-5
evolve_active_component_threshold = 0.1
```

These values are wrapper defaults; all evolve-LoRA arguments can also be overridden from the command line.

Training outputs are written under:

```text
experiments/arithmetic/<model>/<run>/
├── checkpoints/
├── logs/
├── tokenizer/
├── config.json
├── training_args.json
├── wandb_run_id.txt
└── final_model/
```

The `final_model` directory contains the saved evolve-LoRA adapter configuration and adapter weights.

### Arithmetic evaluation

For the standard merged-model arithmetic evaluation pipeline:

```bash
bash scripts/arithmetic_merge_eval.sh /abs/path/to/run_dir
```

The script accepts either a run directory or a direct `final_model` path. Multiple runs can be evaluated in one invocation:

```bash
bash scripts/arithmetic_merge_eval.sh \\
  /abs/path/to/run1 \\
  /abs/path/to/run2
```

**Important:** raw evolve-LoRA adapters are input-conditioned and cannot be exactly merged into static base-model weights. Use the HF inference/evaluation path below for evolve-LoRA adapters.

## Evolve-LoRA inference

### Hugging Face

Use the HF backend for raw evolve-LoRA adapters:

```bash
bash scripts/infer_evolve_lora_hf.sh \\
  mistralai/Mistral-7B-v0.1 \\
  /path/to/run/final_model \\
  'Solve: 17 * 23 =' \\
  --max_new_tokens 128
```

You can also call the Python entrypoint directly:

```bash
python inference_evolve_lora.py \\
  --backend hf \\
  --model mistralai/Mistral-7B-v0.1 \\
  --adapter_path /path/to/run/final_model \\
  --prompt 'Solve: 17 * 23 =' \\
  --max_new_tokens 128
```

Sampling options are available directly from the inference script:

```bash
--do_sample \\
--temperature 0.7 \\
--top_p 0.95
```

`temperature` must be strictly positive. If deterministic generation is desired, omit `--do_sample` rather than setting `--temperature 0`.

For explicit dtype/device control:

```bash
python inference_evolve_lora.py \\
  --backend hf \\
  --model mistralai/Mistral-7B-v0.1 \\
  --adapter_path /path/to/run/final_model \\
  --prompt 'Solve: 17 * 23 =' \\
  --torch_dtype bfloat16 \\
  --device_map auto
```

### vLLM

The custom evolve-LoRA adapter is **not supported as a raw adapter by vLLM**, because its spectral gates depend on the current input. The vLLM backend in `inference_evolve_lora.py` is intended only for a full model directory that is already compatible with vLLM:

```bash
python inference_evolve_lora.py \\
  --backend vllm \\
  --model /path/to/full/model \\
  --prompt 'Solve: 17 * 23 =' \\
  --max_new_tokens 128
```

Do not pass `--adapter_path` with `--backend vllm`.

## Commonsense reasoning

The repository contains the existing Commonsense170K training and evaluation pipeline. The standard commonsense evaluation covers:

- ARC-Challenge
- ARC-Easy
- BoolQ
- HellaSwag
- OpenBookQA
- PIQA
- Social IQa
- WinoGrande

The standard evaluation script is:

```bash
bash scripts/cr_merge_eval.sh
```

Use the files and scripts present in this branch for the exact commonsense training/evaluation configuration. In particular, do not assume that a dedicated `train_cr_evolve_lora.sh` wrapper exists unless it is present in the checkout.

## Command-line options

The main evolve-LoRA controls exposed by `train_arithmetic.py` include:

| Argument | Default | Description |
|---|---:|---|
| `--adapter_type` | `abba` | Adapter type; use `evolve_lora` for evolve-LoRA |
| `--lora_r` | `32` | Maximum evolve-LoRA rank |
| `--lora_alpha` | `16` | Adapter scaling parameter |
| `--lora_dropout` | `0` | Adapter dropout |
| `--evolve_rank_delay_ratio` | `0.15` | Rank-regularization delay ratio; currently inactive in the task-only objective |
| `--evolve_r_min` | `2` | Minimum target effective rank; currently retained as a configuration option |
| `--evolve_beta` | `0.05` | Rank/balance regularization coefficient; currently inactive |
| `--evolve_alpha_max` | `0.01` | Maximum rank-pressure coefficient; currently inactive |
| `--evolve_anneal_k` | `5e-5` | Rank-pressure annealing parameter; currently inactive |
| `--evolve_gate_floor` | `0.05` | Minimum spectral gate value |
| `--evolve_complexity_ema` | `0.9` | EMA smoothing parameter for complexity tracking |
| `--evolve_router_hidden_dim` | `64` | Router hidden dimension configuration |
| `--evolve_active_component_threshold` | `0.1` | Threshold used to count active spectral components |
| `--evolve_active_log_max_layers` | `0` | Maximum number of layers logged individually; `0` logs all layers |
| `--evolve_no_detach_router` | `false` | Allow router gradients through the router input |
| `--batch_size` | `1` | Per-device training batch size |
| `--grad_acc_steps` | `32` | Gradient accumulation steps |
| `--epochs` | `1` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--warmup_ratio` | `0.02` | Warmup ratio |
| `--max_seq_length` | `512` | Maximum sequence length |

## Saving and loading adapters

Training saves an adapter-specific configuration as:

```text
final_model/adapter_config.json
final_model/adapter_model.bin
```

For inference, the base model must be loaded first, then the saved evolve-LoRA configuration and weights are applied. This is handled automatically by `inference_evolve_lora.py`.

Because the adapter is input-conditioned, the base model and evolve-LoRA adapter should be used together during inference; the adapter weights alone are not a standalone language model.

## Original ABBA experiments

The repository also retains the original ABBA PEFT experiments. For arithmetic training:

```bash
bash scripts/train_arithmetic.sh
```

For arithmetic evaluation:

```bash
bash scripts/arithmetic_merge_eval.sh /abs/path/to/run_dir
```

The ABBA implementation and its original paper are separate from the evolve-LoRA path documented above.

## Citation

If you use the original ABBA implementation, please cite:

```bibtex
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

## Notes

- `evolve-LoRA` is **input-conditioned** and therefore cannot be represented by one static LoRA delta.
- Use `--backend hf` when loading a raw evolve-LoRA adapter.
- The current trainer optimizes the task loss only; rank/entropy/balance/orthogonality code is retained for experimentation and logging.
- The repository should be treated as an experimental research codebase; command-line defaults may change as the rank-selection experiments evolve.
