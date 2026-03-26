import json
import os
import sys
from typing import Dict

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from moe_lora import MoELoRAConfig, get_moe_lora_model, load_moe_checkpoint_flexible, load_moe_checkpoint_state_dict


def _normalize_key(key: str) -> str:
    return key[len("model."):] if key.startswith("model.") else key


def maybe_normalize_rank_moe_checkpoint(model_path: str) -> str:
    """Normalize Rank-MoE checkpoint key prefixes in place for eval loader compatibility.

    Our vLLM eval stack expects checkpoint names without a leading ``model.``.
    If a checkpoint is saved with ``model.*`` keys, rewrite to ``*``.
    """
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(index_path) and not os.path.exists(single_path):
        return model_path

    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_data: Dict = json.load(f)
        weight_map = index_data.get("weight_map", {})
        if not isinstance(weight_map, dict):
            return model_path
        keys = list(weight_map.keys())
    else:
        keys = list(load_file(single_path).keys())

    has_model_prefix = any(k.startswith("model.") for k in keys)
    has_plain_prefix = any(not k.startswith("model.") for k in keys)
    if not has_model_prefix or has_plain_prefix:
        return model_path

    print(f"[moe_eval_utils] Normalizing checkpoint key prefix in place: {model_path}")

    if os.path.exists(index_path):
        shard_names = sorted(set(weight_map.values()))
        new_weight_map = {}
        for shard_name in shard_names:
            in_shard = os.path.join(model_path, shard_name)
            tensors = load_file(in_shard)
            normalized = {}
            for k, v in tensors.items():
                nk = _normalize_key(k)
                normalized[nk] = v
                new_weight_map[nk] = shard_name
            tmp_shard = f"{in_shard}.tmp"
            save_file(normalized, tmp_shard)
            os.replace(tmp_shard, in_shard)

        tmp_index = f"{index_path}.tmp"
        with open(tmp_index, "w") as f:
            json.dump({"metadata": index_data.get("metadata", {}), "weight_map": new_weight_map}, f, indent=2)
        os.replace(tmp_index, index_path)
    else:
        tensors = load_file(single_path)
        normalized = {_normalize_key(k): v for k, v in tensors.items()}
        tmp_single = f"{single_path}.tmp"
        save_file(normalized, tmp_single)
        os.replace(tmp_single, single_path)

    return model_path


def model_path_candidates(model_path: str):
    """Candidate model paths to try for evaluation loaders."""
    return [maybe_normalize_rank_moe_checkpoint(model_path)]


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
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        tokenizer_source = tokenizer_path if tokenizer_path else base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_context_len = int(getattr(self.model.config, "max_position_embeddings", 4096))
        if getattr(self.tokenizer, "model_max_length", 0) < self.max_context_len:
            self.tokenizer.model_max_length = self.max_context_len

    def generate(self, prompts, sampling_params):
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_context_len,
        ).to(self.device)
        if "attention_mask" in inputs and inputs["attention_mask"].dtype is not torch.bool:
            inputs["attention_mask"] = inputs["attention_mask"].bool()
        do_sample = float(getattr(sampling_params, "temperature", 0.0)) > 0.0
        gen_kwargs = {
            "max_new_tokens": int(getattr(sampling_params, "max_tokens", 256)),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(getattr(sampling_params, "temperature", 1.0))
            gen_kwargs["top_p"] = float(getattr(sampling_params, "top_p", 1.0))
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        completions = generated[:, prompt_len:]
        return self.tokenizer.batch_decode(completions, skip_special_tokens=True)


def create_generation_backend(model_path: str, tokenizer_path: str | None, tensor_parallel_size: int, backend: str = "auto"):
    if backend == "vllm":
        normalized_path = maybe_normalize_rank_moe_checkpoint(model_path)
        return VLLMBackend(normalized_path, tokenizer_path, tensor_parallel_size)
    if backend == "hf_moe":
        return HFMoEBackend(model_path, tokenizer_path)

    if is_adaptive_moe_checkpoint(model_path):
        print("[moe_eval_utils] Detected adaptive MoE checkpoint; using HF MoE backend for eval.")
        return HFMoEBackend(model_path, tokenizer_path)
    normalized_path = maybe_normalize_rank_moe_checkpoint(model_path)
    return VLLMBackend(normalized_path, tokenizer_path, tensor_parallel_size)
