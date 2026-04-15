import torch
import os
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    Trainer,
)
from abba import ABBAConfig, get_abba_model
from moe_lora import MoELoRAConfig, get_moe_lora_model
from moe_lora import RankMoELoRALayer
from datasets import load_dataset
import numpy as np
from peft import (
    get_peft_model,
    AdaLoraModel,
    AdaLoraConfig,
    TaskType,
    LoraConfig,
    prepare_model_for_kbit_training,
)
from utils.data_utils import *
import argparse
from copy import deepcopy
from tqdm import tqdm

from peft.utils import _get_submodules
from huggingface_hub import snapshot_download

def create_moe_optimizer_param_groups(model, args):
    """Build optimizer groups tuned for MoE-LoRA training stability."""
    lr = float(args.lr)
    router_lr = float(getattr(args, "moe_router_lr", lr))
    lora_weight_decay = float(getattr(args, "moe_lora_weight_decay", args.weight_decay))
    router_weight_decay = float(getattr(args, "moe_router_weight_decay", 0.0))
    no_decay_terms = ("bias", "norm", "ln_", "layernorm")

    lora_decay, lora_no_decay = [], []
    router_decay, router_no_decay = [], []
    other_trainable = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower_name = name.lower()
        is_no_decay = any(term in lower_name for term in no_decay_terms)
        if "router" in lower_name:
            (router_no_decay if is_no_decay else router_decay).append(param)
        elif any(tag in lower_name for tag in ("lora", ".a", ".b", "s_a", "s_b")):
            (lora_no_decay if is_no_decay else lora_decay).append(param)
        else:
            other_trainable.append(param)

    param_groups = []
    if lora_decay:
        param_groups.append({"params": lora_decay, "lr": lr, "weight_decay": lora_weight_decay})
    if lora_no_decay:
        param_groups.append({"params": lora_no_decay, "lr": lr, "weight_decay": 0.0})
    if router_decay:
        param_groups.append({"params": router_decay, "lr": router_lr, "weight_decay": router_weight_decay})
    if router_no_decay:
        param_groups.append({"params": router_no_decay, "lr": router_lr, "weight_decay": 0.0})
    if other_trainable:
        param_groups.append({"params": other_trainable, "lr": lr, "weight_decay": lora_weight_decay})

    return param_groups

def _configure_hf_download(args):
    if getattr(args, "hf_fast_download", False):
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "true")
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        os.environ.setdefault(
            "HF_PARALLEL_LOADING_WORKERS",
            str(getattr(args, "hf_parallel_loading_workers", 8)),
        )

def _resolve_model_source(args):
    if not getattr(args, "hf_preload", False):
        return args.model

    snapshot_kwargs = {
        "repo_id": args.model,
        "resume_download": True,
        "max_workers": getattr(args, "hf_download_workers", 16),
    }
    cache_dir = getattr(args, "hf_cache_dir", None)
    if cache_dir:
        snapshot_kwargs["cache_dir"] = cache_dir
    if getattr(args, "hf_local_files_only", False):
        snapshot_kwargs["local_files_only"] = True

    if getattr(args, "hf_prefer_safetensors", False):
        snapshot_kwargs["ignore_patterns"] = ["*.bin", "*.pth"]
        try:
            return snapshot_download(**snapshot_kwargs)
        except Exception:
            snapshot_kwargs.pop("ignore_patterns", None)

    return snapshot_download(**snapshot_kwargs)

def _get_model_load_kwargs(args):
    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    cache_dir = getattr(args, "hf_cache_dir", None)
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    if getattr(args, "hf_local_files_only", False):
        load_kwargs["local_files_only"] = True
    if getattr(args, "hf_prefer_safetensors", False):
        load_kwargs["use_safetensors"] = True
    return load_kwargs

def _get_tokenizer_load_kwargs(args):
    load_kwargs = {
        "use_fast": True,
        "model_max_length": args.max_seq_length,
        "padding": "max_length",
    }
    cache_dir = getattr(args, "hf_cache_dir", None)
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    if getattr(args, "hf_local_files_only", False):
        load_kwargs["local_files_only"] = True
    return load_kwargs

def create_model_tokenizer_it(args):
    _configure_hf_download(args)
    model_source = _resolve_model_source(args)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            **_get_model_load_kwargs(args),
        )
    except Exception:
        fallback_kwargs = _get_model_load_kwargs(args)
        fallback_kwargs.pop("use_safetensors", None)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            **fallback_kwargs,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        **_get_tokenizer_load_kwargs(args),
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def create_model_tokenizer_cr(args):
    _configure_hf_download(args)
    model_source = _resolve_model_source(args)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            **_get_model_load_kwargs(args),
        )
    except Exception:
        fallback_kwargs = _get_model_load_kwargs(args)
        fallback_kwargs.pop("use_safetensors", None)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            **fallback_kwargs,
        )
    
    if "llama" in args.model:

        if "Llama-3" in args.model:
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                **_get_tokenizer_load_kwargs(args),
            )
        else:
            tokenizer = LlamaTokenizer.from_pretrained(
                model_source,
                **_get_tokenizer_load_kwargs(args),
            )

    else:

        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            **_get_tokenizer_load_kwargs(args),
        )

    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"

    return model, tokenizer


def create_peft_model_it_abba(model, args):

    abba_config = ABBAConfig(
        r1=args.lora_r,                     
        r2=args.lora_r,                     
        alpha1=args.lora_alpha,                 
        alpha2=args.lora_alpha,                 
        dropout=args.lora_dropout,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_abba_model(model, abba_config)

    return model, abba_config

def create_peft_model_cr_abba(model, args):

    abba_config = ABBAConfig(
        r1=args.lora_r,                     
        r2=args.lora_r,                     
        alpha1=args.lora_alpha,                 
        alpha2=args.lora_alpha,                 
        dropout=args.lora_dropout,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_abba_model(model, abba_config)

    return model, abba_config


def _parse_moe_experts_config(rank_csv: str):
    ranks = [int(rank.strip()) for rank in rank_csv.split(",") if rank.strip()]
    return [{"rank": rank} for rank in ranks]


def _build_moe_experts_config(args):
    # Preferred path: explicit r_max only.
    if getattr(args, "moe_r_max", 0) > 0:
        return [{"rank": int(args.moe_r_max)}]

    # Backward-compat fallback for older configs.
    rank_csv = getattr(args, "moe_rank_components", "") or getattr(args, "moe_expert_ranks", "")
    if rank_csv:
        return _parse_moe_experts_config(rank_csv)
    raise ValueError("Set --moe_r_max (preferred) or provide rank components.")


def create_peft_model_it_moe_lora(model, args):
    router_hidden_dim = args.moe_router_hidden_dim if args.moe_router_hidden_dim > 0 else None
    moe_config = MoELoRAConfig(
        experts_config=_build_moe_experts_config(args),
        r_max=args.moe_r_max if args.moe_r_max > 0 else None,
        top_k=args.moe_top_k,
        router_hidden_dim=router_hidden_dim,
        router_norm_type=args.moe_router_norm_type,
        router_activation=args.moe_router_activation,
        entropy_loss_weight=args.moe_entropy_loss_weight,
        load_balance_loss_weight=args.moe_load_balance_loss_weight,
        mask_init_strategy=args.moe_mask_init_strategy,
        mask_init_value=args.moe_mask_init_value,
        mask_init_std=args.moe_mask_init_std,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        freeze_base=True,
    )
    model = get_moe_lora_model(model, moe_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    trainable_params = sum(p.requires_grad for p in model.parameters())
    if trainable_params == 0:
        raise RuntimeError("MoE-LoRA setup produced zero trainable parameters.")
    return model, moe_config


def create_peft_model_cr_moe_lora(model, args):
    router_hidden_dim = args.moe_router_hidden_dim if args.moe_router_hidden_dim > 0 else None
    moe_config = MoELoRAConfig(
        experts_config=_build_moe_experts_config(args),
        r_max=args.moe_r_max if args.moe_r_max > 0 else None,
        top_k=args.moe_top_k,
        router_hidden_dim=router_hidden_dim,
        router_norm_type=args.moe_router_norm_type,
        router_activation=args.moe_router_activation,
        entropy_loss_weight=args.moe_entropy_loss_weight,
        load_balance_loss_weight=args.moe_load_balance_loss_weight,
        mask_init_strategy=args.moe_mask_init_strategy,
        mask_init_value=args.moe_mask_init_value,
        mask_init_std=args.moe_mask_init_std,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        freeze_base=True,
    )
    model = get_moe_lora_model(model, moe_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    trainable_params = sum(p.requires_grad for p in model.parameters())
    if trainable_params == 0:
        raise RuntimeError("MoE-LoRA setup produced zero trainable parameters.")
    return model, moe_config


class MoEAuxLossTrainer(Trainer):
    """Trainer with optional MoE routing regularizers."""

    def __init__(
        self,
        *args,
        moe_entropy_loss_weight: float = 0.0,
        moe_load_balance_loss_weight: float = 0.0,
        moe_aux_loss_cap: float = 0.0,
        moe_aux_warmup_ratio: float = 0.0,
        moe_aux_stop_ratio: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.moe_entropy_loss_weight = float(moe_entropy_loss_weight)
        self.moe_load_balance_loss_weight = float(moe_load_balance_loss_weight)
        self.moe_aux_loss_cap = float(moe_aux_loss_cap)
        self.moe_aux_warmup_ratio = float(moe_aux_warmup_ratio)
        self.moe_aux_stop_ratio = float(moe_aux_stop_ratio)
        self._moe_layers = [m for m in self.model.modules() if isinstance(m, RankMoELoRALayer)]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            loss = outputs.loss

        max_steps = max(int(getattr(self.state, "max_steps", 0) or 0), 1)
        progress = float(self.state.global_step) / float(max_steps)
        aux_scale = 1.0
        if self.moe_aux_stop_ratio > 0.0 and progress >= self.moe_aux_stop_ratio:
            aux_scale = 0.0
        elif self.moe_aux_warmup_ratio > 0.0 and progress < self.moe_aux_warmup_ratio:
            aux_scale = max(progress / max(self.moe_aux_warmup_ratio, 1e-8), 0.0)

        effective_entropy_weight = self.moe_entropy_loss_weight * aux_scale
        effective_load_balance_weight = self.moe_load_balance_loss_weight * aux_scale

        entropy_loss = None
        load_balance_loss = None
        if effective_entropy_weight != 0.0 or effective_load_balance_weight != 0.0:
            entropy_terms = []
            load_balance_terms = []
            for module in self._moe_layers:
                if effective_entropy_weight != 0.0 and module._last_rank_entropy_loss is not None:
                    entropy_terms.append(module._last_rank_entropy_loss)
                if effective_load_balance_weight != 0.0 and module._last_load_balance_loss is not None:
                    load_balance_terms.append(module._last_load_balance_loss)

            if entropy_terms:
                entropy_loss = torch.stack(entropy_terms).mean()
                loss = loss + effective_entropy_weight * entropy_loss
            if load_balance_terms:
                load_balance_loss = torch.stack(load_balance_terms).mean()
                loss = loss + effective_load_balance_weight * load_balance_loss
            if self.moe_aux_loss_cap > 0.0 and (entropy_loss is not None or load_balance_loss is not None):
                max_allowed = outputs["loss"].detach().abs() * self.moe_aux_loss_cap
                total_aux = (loss - outputs["loss"]).abs()
                if total_aux > max_allowed:
                    aux_clip_scale = (max_allowed / total_aux).detach()
                    loss = outputs["loss"] + (loss - outputs["loss"]) * aux_clip_scale

        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            log_payload = {"base_loss": float(outputs["loss"].detach().float())}
            if entropy_loss is not None:
                log_payload["moe_entropy_loss"] = float(entropy_loss.detach().float())
            if load_balance_loss is not None:
                log_payload["moe_load_balance_loss"] = float(load_balance_loss.detach().float())
            if self.moe_entropy_loss_weight != 0.0 or self.moe_load_balance_loss_weight != 0.0:
                log_payload["moe_aux_scale"] = float(aux_scale)
                log_payload["moe_entropy_loss_weight_effective"] = float(effective_entropy_weight)
                log_payload["moe_load_balance_loss_weight_effective"] = float(effective_load_balance_weight)
            self.log(log_payload)

        return (loss, outputs) if return_outputs else loss
