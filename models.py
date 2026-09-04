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
)
from abba import ABBAConfig, get_abba_model
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

# evolve-LoRA helpers are kept separate from ABBA so existing ABBA code paths remain unchanged.
from evolve_lora import EvolveLoRAConfig, apply_evolve_lora
from sora import SoRAConfig, apply_sora


def create_peft_model_it_evolve_lora(model, args):
    config = EvolveLoRAConfig(
        r_max=args.lora_r,
        r_min=args.evolve_r_min,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        gate_floor=args.evolve_gate_floor,
        detach_router_input=not args.evolve_no_detach_router,
        beta=args.evolve_beta,
        evolve_rank_delay_ratio=getattr(args, "evolve_rank_delay_ratio", 0.15),
        alpha_max=args.evolve_alpha_max,
        anneal_k=args.evolve_anneal_k,
        complexity_ema=args.evolve_complexity_ema,
        router_hidden_dim=getattr(args, "evolve_router_hidden_dim", 64),
        ortho_weight=args.ortho_weight,
        active_component_threshold=args.evolve_active_component_threshold,
        active_log_max_layers=args.evolve_active_log_max_layers,
        active_log_seed=args.seed,
    )
    return apply_evolve_lora(model, config), config


def create_peft_model_cr_evolve_lora(model, args):
    return create_peft_model_it_evolve_lora(model, args)


def _target_modules():
    return ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

def create_peft_model_it_lora(model, args):
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=_target_modules(),
    )
    return get_peft_model(model, config), config

def create_peft_model_cr_lora(model, args):
    return create_peft_model_it_lora(model, args)

def create_peft_model_it_adalora(model, args):
    config = AdaLoraConfig(
        init_r=args.lora_r,
        target_r=getattr(args, "adalora_target_r", max(1, args.lora_r // 2)),
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        beta1=getattr(args, "adalora_beta1", 0.85),
        beta2=getattr(args, "adalora_beta2", 0.85),
        tinit=getattr(args, "adalora_tinit", 200),
        tfinal=getattr(args, "adalora_tfinal", 1000),
        deltaT=getattr(args, "adalora_deltaT", 10),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=_target_modules(),
    )
    return get_peft_model(model, config), config

def create_peft_model_cr_adalora(model, args):
    return create_peft_model_it_adalora(model, args)

def create_peft_model_it_sora(model, args):
    config = SoRAConfig(
        rmax=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=_target_modules(),
        gate_lr=getattr(args, "sora_gate_lr", 0.1),
        lambda_sparsity=getattr(args, "sora_lambda_sparsity", 0.1),
    )
    return apply_sora(model, config), config

def create_peft_model_cr_sora(model, args):
    return create_peft_model_it_sora(model, args)

def create_adapter_model_it(model, args):
    if args.adapter_type == "evolve_lora":
        return create_peft_model_it_evolve_lora(model, args)
    if args.adapter_type == "abba":
        return create_peft_model_it_abba(model, args)
    if args.adapter_type == "lora":
        return create_peft_model_it_lora(model, args)
    if args.adapter_type == "adalora":
        return create_peft_model_it_adalora(model, args)
    if args.adapter_type == "sora":
        return create_peft_model_it_sora(model, args)
    raise ValueError(f"Unsupported adapter_type: {args.adapter_type}")

def create_adapter_model_cr(model, args):
    if args.adapter_type == "evolve_lora":
        return create_peft_model_cr_evolve_lora(model, args)
    if args.adapter_type == "abba":
        return create_peft_model_cr_abba(model, args)
    if args.adapter_type == "lora":
        return create_peft_model_cr_lora(model, args)
    if args.adapter_type == "adalora":
        return create_peft_model_cr_adalora(model, args)
    if args.adapter_type == "sora":
        return create_peft_model_cr_sora(model, args)
    raise ValueError(f"Unsupported adapter_type: {args.adapter_type}")
