# Rank Selection / Evolve-LoRA

Code for efficient fine-tuning experiments with **input-conditioned spectral LoRA (evolve-LoRA)** and the original ABBA adapter implementation.

> **Branch:** `evolve_rank`
>
> The `evolve_rank` branch contains the current evolve-LoRA implementation and model-specific training/evaluation utilities described below.

## Overview

The evolve-LoRA adapter parameterizes the weight update as

$$
\Delta W(x) = U\,\mathrm{diag}(\lambda(x))\,V^T,
$$

where `U` and `V` are learned low-rank factors and the spectral coefficients `lambda(x)` are predicted from the input by a lightweight router. This makes the effective adapter rank **input-dependent** rather than fixed for every example.

The current implementation uses a **standard softmax mixture-of-experts router**:

$$
\lambda(x) = \mathrm{softmax}(R(x)).
$$

In the current model-specific experiments, `evolve_gate_floor=0`, so there is **no gate floor or additive offset**. The router is simply `softmax(R(x))` over the `r_max` spectral components.

The adapter is therefore dynamic at inference time and cannot be exactly merged into a single static weight matrix.

![intro-fig](assets/abba_github.jpg)

## Repository structure

Important files for evolve-LoRA:

```text
evolve_lora.py                 # Evolve-LoRA module, router, rank metrics, Trainer
train_arithmetic.py             # Arithmetic training entrypoint
train_cr.py                     # Commonsense reasoning training entrypoint
inference_evolve_lora.py        # Hugging Face / vLLM inference entrypoint
scripts/
  train_arithmetic_evolve_lora.sh
  train_arithmetic_evolve_lora_Mistral7B.sh
  train_arithmetic_evolve_lora_Gemma2-9B.sh
  train_cr_evolve_lora_Llama1B.sh
  train_cr_evolve_lora_Llama3B.sh
  arithmetic_evolve_lora_hf_eval_Mistral7B.sh
  arithmetic_evolve_lora_hf_eval_Gemma2-9B.sh
  cr_evolve_lora_hf_eval.sh
  infer_evolve_lora_hf.sh
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

For a base linear layer `W`, evolve-LoRA keeps the base weights frozen and adds

$$
y = Wx + \left(xV \odot \lambda(x)\right)U^T \cdot \frac{\alpha}{r_{\max}},
$$

where:

- `U ∈ R^(d_out × r_max)` is the output factor.
- `V ∈ R^(d_in × r_max)` is the input factor.
- `r_max` is the maximum adapter rank.
- `lambda(x)` is produced by the input-conditioned softmax router.
- `alpha / r_max` is the adapter scaling factor.

The router is a lightweight linear projection from the layer input to `r_max` spectral components followed by softmax. There is no hidden MLP in the current router implementation. Router inputs can optionally be detached from the computation graph with `--evolve_no_detach_router` controlling this behavior.

The model-specific evolve-LoRA experiments use `--evolve_gate_floor 0`, giving exactly

$$
\lambda(x) = \mathrm{softmax}(R(x)).
$$

### Rank statistics

The implementation reports an entropy-based effective rank:

$$
r_{\mathrm{eff}} = \exp\left(-\sum_i p_i\log p_i\right),
$$

where the current implementation uses the spectral coefficients directly as `p_i`. Because the current router is a softmax, these coefficients form a normalized distribution.

It also records the number of active spectral components above `--evolve_active_component_threshold`. These metrics are logged under the `evolve/*` namespace in Weights & Biases.

### Current loss behavior

The current `evolve_rank` implementation uses the **task loss only** for optimization:

$$
\mathcal{L} = \mathcal{L}_{task}.
$$

The rank, entropy, balance, and orthogonality utilities remain available in `evolve_lora.py` for experimentation, but the corresponding regularization terms are currently disabled in `EvolveLoRATrainer.compute_loss`. The reported effective rank and active-component statistics are monitoring metrics rather than optimization losses.

## Experiments

The evolve-LoRA experiments are organized by model. **Each model has its own training launcher and its own evaluation launcher, except Llama 3.2 1B and Llama 3.2 3B, which share the same commonsense HF evaluation script.**

| Model | Task | Training script | Evaluation script |
|---|---|---|---|
| Mistral 7B | Arithmetic | `scripts/train_arithmetic_evolve_lora_Mistral7B.sh` | `scripts/arithmetic_evolve_lora_hf_eval_Mistral7B.sh` |
| Gemma 2 9B | Arithmetic | `scripts/train_arithmetic_evolve_lora_Gemma2-9B.sh` | `scripts/arithmetic_evolve_lora_hf_eval_Gemma2-9B.sh` |
| Llama 3.2 1B | Commonsense | `scripts/train_cr_evolve_lora_Llama1B.sh` | `scripts/cr_evolve_lora_hf_eval.sh` |
| Llama 3.2 3B | Commonsense | `scripts/train_cr_evolve_lora_Llama3B.sh` | `scripts/cr_evolve_lora_hf_eval.sh` |

### Mistral 7B — arithmetic

Training uses the model-specific launcher:

```bash
bash scripts/train_arithmetic_evolve_lora_Mistral7B.sh
```

The launcher is configured for `mistralai/Mistral-7B-v0.1`, with maximum rank 22, `lora_alpha=44`, task-only optimization, and the evolve-LoRA settings used for the Mistral experiment. Additional arguments are forwarded to `train_arithmetic.py`.

Evaluate the saved adapter directly with Hugging Face:

```bash
bash scripts/arithmetic_evolve_lora_hf_eval_Mistral7B.sh \\
  /path/to/run
```

This evaluates GSM8K and MATH without attempting to merge the input-conditioned adapter.

### Gemma 2 9B — arithmetic

Training uses the dedicated Gemma launcher:

```bash
bash scripts/train_arithmetic_evolve_lora_Gemma2-9B.sh
```

The launcher is configured for `google/gemma-2-9b`, with maximum rank 22, `lora_alpha=44`, task-only optimization, and the evolve-LoRA settings used for the Gemma experiment. Additional arguments are forwarded to `train_arithmetic.py`.

Evaluate with:

```bash
bash scripts/arithmetic_evolve_lora_hf_eval_Gemma2-9B.sh \\
  /path/to/run
```

This evaluates GSM8K and MATH using Hugging Face adapter inference.

### Llama 3.2 1B — commonsense reasoning

Training uses the dedicated Llama 1B launcher:

```bash
bash scripts/train_cr_evolve_lora_Llama1B.sh
```

The launcher uses `meta-llama/Llama-3.2-1B` and the evolve-LoRA configuration for the 1B commonsense experiment.

Llama 1B and Llama 3B use the **same evaluation script**. Select the model through the `MODEL` environment variable:

```bash
MODEL=meta-llama/Llama-3.2-1B \\
  bash scripts/cr_evolve_lora_hf_eval.sh /path/to/run
```

The evaluation covers ARC-Challenge, ARC-Easy, BoolQ, HellaSwag, OpenBookQA, PIQA, Social IQa, and WinoGrande.

### Llama 3.2 3B — commonsense reasoning

Training uses the dedicated Llama 3B launcher:

```bash
bash scripts/train_cr_evolve_lora_Llama3B.sh
```

The launcher uses `meta-llama/Llama-3.2-3B` and the evolve-LoRA configuration for the 3B commonsense experiment.

The same shared evaluation launcher is used for Llama 3B:

```bash
MODEL=meta-llama/Llama-3.2-3B \\
  bash scripts/cr_evolve_lora_hf_eval.sh /path/to/run
```

### Generic arithmetic launcher

A generic arithmetic wrapper is also available when you want to override the model/configuration from the command line:

```bash
bash scripts/train_arithmetic_evolve_lora.sh \\
  --model mistralai/Mistral-7B-v0.1 \\
  --dataset_split 'train[:50000]'
```

For reproducible model-specific experiments, prefer the dedicated launchers in the table above.

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
| `--evolve_gate_floor` | `0.05` | Gate floor configuration; model-specific experiments set this to `0` |
| `--evolve_complexity_ema` | `0.9` | EMA smoothing parameter for complexity tracking |
| `--evolve_router_hidden_dim` | `64` | Retained router hidden-dimension configuration; the current router itself is a single linear projection |
| `--evolve_active_component_threshold` | `0.1` | Threshold used to count active spectral components |
| `--evolve_active_log_max_layers` | `0` | Maximum number of layers logged individually; `0` logs all layers |
| `--evolve_no_detach_router` | `false` | Allow router gradients through the router input |
| `--batch_size` | `1` | Per-device training batch size |
| `--grad_acc_steps` | `32` | Gradient accumulation steps |
| `--epochs` | `1` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--warmup_ratio` | `0.02` | Warmup ratio |
| `--max_seq_length` | `512` | Maximum sequence length |

For the model-specific experiment launchers, the important router setting is:

```bash
--evolve_gate_floor 0
```

which gives the standard MoE-style softmax router

$$
\lambda(x) = \mathrm{softmax}(R(x)).
$$

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
- The current router is a standard softmax router: `softmax(R(x))`.
- Model-specific experiments use `--evolve_gate_floor 0`; there is no gate-floor offset in those experiments.
- Use `--backend hf` when loading a raw evolve-LoRA adapter.
- The current trainer optimizes the task loss only; rank/entropy/balance/orthogonality code is retained for experimentation and logging.
- The repository should be treated as an experimental research codebase; command-line defaults may change as the rank-selection experiments evolve.
