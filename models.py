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
from moe_lora import MoELoRAConfig, get_moe_lora_model
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


def _parse_moe_experts_config(rank_csv: str):
    ranks = [int(rank.strip()) for rank in rank_csv.split(",") if rank.strip()]
    return [{"rank": rank} for rank in ranks]


def create_peft_model_it_moe_lora(model, args):
    router_hidden_dim = args.moe_router_hidden_dim if args.moe_router_hidden_dim > 0 else None
    moe_config = MoELoRAConfig(
        experts_config=_parse_moe_experts_config(args.moe_expert_ranks),
        top_k=args.moe_top_k,
        router_hidden_dim=router_hidden_dim,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        freeze_base=True,
    )
    model = get_moe_lora_model(model, moe_config)
    return model, moe_config


def create_peft_model_cr_moe_lora(model, args):
    router_hidden_dim = args.moe_router_hidden_dim if args.moe_router_hidden_dim > 0 else None
    moe_config = MoELoRAConfig(
        experts_config=_parse_moe_experts_config(args.moe_expert_ranks),
        top_k=args.moe_top_k,
        router_hidden_dim=router_hidden_dim,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        freeze_base=True,
    )
    model = get_moe_lora_model(model, moe_config)
    return model, moe_config
