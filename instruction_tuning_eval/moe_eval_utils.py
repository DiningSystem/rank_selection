import json
import os
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def _read_json_if_exists(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _resolve_moe_hparams(model_path: str, fallback_r_max: int = 32, fallback_top_k: int = 1, fallback_router_hidden: int = 128):
    state_dict = load_moe_checkpoint_state_dict(model_path)
    r_max = int(fallback_r_max)
    router_hidden_dim = int(fallback_router_hidden)
    for k, v in state_dict.items():
        if k.endswith(".A") and getattr(v, "ndim", 0) >= 2:
            r_max = int(v.shape[0])
            break
    for k, v in state_dict.items():
        if k.endswith("router.net.0.weight") and getattr(v, "ndim", 0) >= 2:
            router_hidden_dim = int(v.shape[0])
            break

    top_k = int(fallback_top_k)
    config_paths = [
        os.path.join(model_path, "config.json"),
        os.path.join(os.path.dirname(model_path), "config.json"),
    ]
    for config_path in config_paths:
        cfg = _read_json_if_exists(config_path)
        if "moe_top_k" in cfg:
            top_k = int(cfg["moe_top_k"])
            break
    top_k = int(os.getenv("HF_MOE_TOP_K", str(top_k)))
    router_hidden_dim = int(os.getenv("HF_MOE_ROUTER_HIDDEN_DIM", str(router_hidden_dim)))
    return r_max, top_k, router_hidden_dim


class VLLMBackend:
    def __init__(self, model_path: str, tokenizer_path: str | None, tensor_parallel_size: int):
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "vLLM is required for backend='vllm'. Install vllm or use backend='hf_moe' for adaptive MoE checkpoints."
            ) from exc
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path if tokenizer_path else model_path,
            tensor_parallel_size=tensor_parallel_size,
        )

    def generate(self, prompts, sampling_params):
        try:
            from vllm import SamplingParams as VLLMSamplingParams
        except ImportError as exc:
            raise ImportError(
                "vLLM SamplingParams is unavailable. Install vllm or switch backend to 'hf_moe'."
            ) from exc
        if not isinstance(sampling_params, VLLMSamplingParams):
            sampling_params = VLLMSamplingParams(
                temperature=float(getattr(sampling_params, "temperature", 0.0)),
                top_p=float(getattr(sampling_params, "top_p", 1.0)),
                top_k=int(getattr(sampling_params, "top_k", -1)),
                max_tokens=int(getattr(sampling_params, "max_tokens", 256)),
                stop=list(getattr(sampling_params, "stop", []) or []),
            )
        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


class HFMoEBackend:
    def __init__(self, model_path: str, tokenizer_path: str | None):
        print(f"[moe_eval_utils] Initializing HFMoEBackend with model_path={model_path}")
        base_model_name = _resolve_base_model_name(model_path)
        r_max, moe_top_k, router_hidden_dim = _resolve_moe_hparams(model_path)
        print(f"[moe_eval_utils] Resolved MoE hparams: r_max={r_max}, top_k={moe_top_k}, router_hidden_dim={router_hidden_dim}")
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
            top_k=moe_top_k,
            router_hidden_dim=router_hidden_dim,
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
        # Important for batched generation when pad_token_id==eos_token_id:
        # right-padding can place eos at sequence end for shorter prompts and trigger
        # premature stop. Left-padding keeps the final token as real prompt content.
        self.tokenizer.padding_side = "left"
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
                # Accuracy-first fallback: keep context/output budgets unchanged.
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
                raise RuntimeError(
                    "CUDA OOM while generating a single prompt in HF MoE backend even after "
                    "retrying with use_cache=False and full generation budgets preserved. "
                    "For deterministic eval correctness, this path no longer auto-truncates "
                    "input/new-token limits."
                ) from exc
            mid = max(1, len(prompts) // 2)
            left = self.generate(prompts[:mid], sampling_params)
            right = self.generate(prompts[mid:], sampling_params)
            return left + right


def create_generation_backend(model_path: str, tokenizer_path: str | None, tensor_parallel_size: int, backend: str = "auto"):
    if backend == "vllm":
        print("[moe_eval_utils] Using generation backend: vllm (forced)")
        return VLLMBackend(model_path, tokenizer_path, tensor_parallel_size)
    if backend == "hf_moe":
        print("[moe_eval_utils] Using generation backend: hf_moe (forced)")
        return HFMoEBackend(model_path, tokenizer_path)

    if is_adaptive_moe_checkpoint(model_path):
        print("[moe_eval_utils] Detected adaptive MoE checkpoint; using HF MoE backend for eval.")
        return HFMoEBackend(model_path, tokenizer_path)
    print("[moe_eval_utils] Using generation backend: vllm (auto)")
    return VLLMBackend(model_path, tokenizer_path, tensor_parallel_size)
