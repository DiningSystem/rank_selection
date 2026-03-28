import json
import os
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from moe_lora import MoELoRAConfig, get_moe_lora_model, load_moe_checkpoint_flexible, load_moe_checkpoint_state_dict


def model_path_candidates(model_path: str):
    """Candidate model paths to try for evaluation loaders."""
    return [model_path]


def _checkpoint_keys(model_path: str):
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            data = json.load(f)
        return set(data.get("weight_map", {}).keys())
    if os.path.exists(single_path):
        return set(load_file(single_path).keys())
    return set()


def is_adaptive_moe_checkpoint(model_path: str) -> bool:
    keys = _checkpoint_keys(model_path)
    has_rank_components = any(k.endswith(".A") or k.endswith(".B") for k in keys)
    has_base_submodule = any(".base.weight" in k for k in keys)
    return has_rank_components and has_base_submodule


def _resolve_base_model_name(model_path: str) -> str:
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    base = config.get("_name_or_path")
    if not base:
        raise ValueError("Missing _name_or_path in config.json; cannot resolve base model for MoE eval.")
    return base


def _infer_r_max(model_path: str, fallback: int = 32) -> int:
    state_dict = load_moe_checkpoint_state_dict(model_path)
    for k, v in state_dict.items():
        if k.endswith(".A") and getattr(v, "ndim", 0) >= 2:
            return int(v.shape[0])
    return int(fallback)


class VLLMBackend:
    def __init__(self, model_path: str, tokenizer_path: str | None, tensor_parallel_size: int):
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path if tokenizer_path else model_path,
            tensor_parallel_size=tensor_parallel_size,
        )

    def generate(self, prompts, sampling_params):
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


class HFMoEBackend:
    def __init__(self, model_path: str, tokenizer_path: str | None):
        base_model_name = _resolve_base_model_name(model_path)
        r_max = _infer_r_max(model_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
        )
        moe_config = MoELoRAConfig(
            experts_config=[{"rank": r_max}],
            r_max=r_max,
            top_k=1,
            router_hidden_dim=128,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            freeze_base=True,
        )
        model = get_moe_lora_model(base_model, moe_config)
        load_moe_checkpoint_flexible(model, model_path, strict=False)
        self.model = model.eval().to(self.device)
        # KV cache is critical for decode-time throughput in autoregressive generation.
        # Keep it enabled by default and allow opting out via env var for low-memory GPUs.
        self.use_cache = os.getenv("HF_MOE_USE_CACHE", "1") == "1"
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = self.use_cache
        tokenizer_source = tokenizer_path if tokenizer_path else base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_context_len = int(getattr(self.model.config, "max_position_embeddings", 4096))
        if getattr(self.tokenizer, "model_max_length", 0) < self.max_context_len:
            self.tokenizer.model_max_length = self.max_context_len

    def _generate_once(self, prompts, sampling_params, max_input_len=None, max_new_tokens=None, use_cache=None):
        input_len = self.max_context_len if max_input_len is None else int(max_input_len)
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=input_len,
        ).to(self.device)
        if "attention_mask" in inputs and inputs["attention_mask"].dtype is not torch.bool:
            inputs["attention_mask"] = inputs["attention_mask"].bool()
        do_sample = float(getattr(sampling_params, "temperature", 0.0)) > 0.0
        output_len = int(getattr(sampling_params, "max_tokens", 256)) if max_new_tokens is None else int(max_new_tokens)
        use_cache_flag = self.use_cache if use_cache is None else bool(use_cache)
        gen_kwargs = {
            "max_new_tokens": output_len,
            "do_sample": do_sample,
            "use_cache": use_cache_flag,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(getattr(sampling_params, "temperature", 1.0))
            gen_kwargs["top_p"] = float(getattr(sampling_params, "top_p", 1.0))
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        completions = generated[:, prompt_len:].cpu()
        decoded = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
        stop_tokens = getattr(sampling_params, "stop", None)
        if stop_tokens:
            processed = []
            for text in decoded:
                cut = len(text)
                for stop in stop_tokens:
                    idx = text.find(stop)
                    if idx != -1:
                        cut = min(cut, idx)
                processed.append(text[:cut])
            decoded = processed
        del inputs, generated, completions
        return decoded

    def generate(self, prompts, sampling_params):
        try:
            return self._generate_once(prompts, sampling_params)
        except RuntimeError as exc:
            is_oom = "out of memory" in str(exc).lower()
            if not is_oom:
                raise
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            if len(prompts) <= 1:
                prompt = prompts[0]
                base_new_tokens = int(getattr(sampling_params, "max_tokens", 256))
                # First fallback: keep generation length unchanged, only disable cache.
                try:
                    if self.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    return self._generate_once(
                        [prompt],
                        sampling_params,
                        max_input_len=self.max_context_len,
                        max_new_tokens=base_new_tokens,
                        use_cache=False,
                    )
                except RuntimeError as nested_exc:
                    if "out of memory" not in str(nested_exc).lower():
                        raise
                # Second fallback: gradually reduce input length while preserving output budget.
                fallback_input_lens = [
                    max(2048, self.max_context_len // 2),
                    max(1024, self.max_context_len // 4),
                    max(512, self.max_context_len // 8),
                ]
                for max_in in fallback_input_lens:
                    try:
                        if self.device.startswith("cuda"):
                            torch.cuda.empty_cache()
                        return self._generate_once(
                            [prompt],
                            sampling_params,
                            max_input_len=max_in,
                            max_new_tokens=base_new_tokens,
                            use_cache=False,
                        )
                    except RuntimeError as nested_exc:
                        if "out of memory" not in str(nested_exc).lower():
                            raise
                # Final fallback: reduce new tokens only as a last resort.
                fallback_new_tokens = [max(256, base_new_tokens // 2), max(128, base_new_tokens // 4), max(64, base_new_tokens // 8)]
                for max_out in fallback_new_tokens:
                    try:
                        if self.device.startswith("cuda"):
                            torch.cuda.empty_cache()
                        return self._generate_once(
                            [prompt],
                            sampling_params,
                            max_input_len=max(512, self.max_context_len // 8),
                            max_new_tokens=max_out,
                            use_cache=False,
                        )
                    except RuntimeError as nested_exc:
                        if "out of memory" not in str(nested_exc).lower():
                            raise
                raise RuntimeError(
                    "CUDA OOM while generating a single prompt in HF MoE backend even after "
                    "automatic backoff on input length and max_new_tokens."
                ) from exc
            mid = max(1, len(prompts) // 2)
            left = self.generate(prompts[:mid], sampling_params)
            right = self.generate(prompts[mid:], sampling_params)
            return left + right


def create_generation_backend(model_path: str, tokenizer_path: str | None, tensor_parallel_size: int, backend: str = "auto"):
    if backend == "vllm":
        return VLLMBackend(model_path, tokenizer_path, tensor_parallel_size)
    if backend == "hf_moe":
        return HFMoEBackend(model_path, tokenizer_path)

    if is_adaptive_moe_checkpoint(model_path):
        print("[moe_eval_utils] Detected adaptive MoE checkpoint; using HF MoE backend for eval.")
        return HFMoEBackend(model_path, tokenizer_path)
    return VLLMBackend(model_path, tokenizer_path, tensor_parallel_size)
