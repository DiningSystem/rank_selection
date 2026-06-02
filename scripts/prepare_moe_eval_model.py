import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from moe_lora import MoELoRAConfig, RankMoELoRALayer, get_moe_lora_model, load_moe_checkpoint_state_dict, load_moe_checkpoint_flexible


TARGET_MODULES = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]


def _resolve_base_model(checkpoint_dir: str, explicit_base_model: str | None) -> str:
    if explicit_base_model:
        return explicit_base_model
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json under {checkpoint_dir}; pass --base_model explicitly.")
    with open(config_path, "r") as f:
        config = json.load(f)
    base = config.get("_name_or_path")
    if not base:
        raise ValueError("Could not infer base model from config.json; pass --base_model explicitly.")
    return base


def _infer_r_max(checkpoint_dir: str, fallback_r_max: int) -> int:
    state_dict = load_moe_checkpoint_state_dict(checkpoint_dir)
    for key, value in state_dict.items():
        if key.endswith(".A") and value.ndim >= 2:
            return int(value.shape[0])
    return int(fallback_r_max)


def _infer_router_hidden_dim(checkpoint_dir: str, fallback_hidden: int) -> int:
    state_dict = load_moe_checkpoint_state_dict(checkpoint_dir)
    for key, value in state_dict.items():
        # Router block is LayerNorm -> Linear(d_model, hidden) -> GELU -> Linear(hidden, r_max).
        if key.endswith("router.net.1.weight") and value.ndim >= 2:
            return int(value.shape[0])
        if key.endswith("router.net.3.weight") and value.ndim >= 2:
            return int(value.shape[1])
    return int(fallback_hidden)


def _moe_config_paths(checkpoint_dir: str):
    return [
        os.path.join(checkpoint_dir, "config.json"),
        os.path.join(os.path.dirname(checkpoint_dir), "config.json"),
    ]


def _read_json_if_exists(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _read_first_config_value(checkpoint_dir: str, key: str, default):
    for config_path in _moe_config_paths(checkpoint_dir):
        cfg = _read_json_if_exists(config_path)
        if key in cfg:
            return cfg[key]
    return default


def _resolve_moe_hparams(checkpoint_dir: str, args):
    return {
        "r_max": _infer_r_max(checkpoint_dir, args.moe_r_max),
        "top_k": int(_read_first_config_value(checkpoint_dir, "moe_top_k", args.moe_top_k)),
        "router_hidden_dim": _infer_router_hidden_dim(checkpoint_dir, args.moe_router_hidden_dim),
        "router_norm_type": str(_read_first_config_value(checkpoint_dir, "moe_router_norm_type", args.moe_router_norm_type)),
        "router_activation": str(_read_first_config_value(checkpoint_dir, "moe_router_activation", args.moe_router_activation)),
        "mask_init_strategy": str(_read_first_config_value(checkpoint_dir, "moe_mask_init_strategy", args.moe_mask_init_strategy)),
        "mask_init_value": float(_read_first_config_value(checkpoint_dir, "moe_mask_init_value", args.moe_mask_init_value)),
        "mask_init_std": float(_read_first_config_value(checkpoint_dir, "moe_mask_init_std", args.moe_mask_init_std)),
        "router_temperature": float(_read_first_config_value(checkpoint_dir, "moe_router_temperature_end", args.moe_router_temperature)),
    }


def _set_moe_router_temperature(model, temperature: float) -> int:
    updated = 0
    for module in model.modules():
        if isinstance(module, RankMoELoRALayer):
            module.set_routing_temperature(temperature)
            updated += 1
    return updated


def main():
    parser = argparse.ArgumentParser(description="Prepare eval-ready full MoE model from base model + MoE checkpoint.")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to MoE checkpoint directory (e.g., run/final_model)")
    parser.add_argument("--output_dir", required=True, help="Where to save eval-ready full model")
    parser.add_argument("--base_model", default=None, help="Optional HF base model name/path override")
    parser.add_argument("--moe_r_max", type=int, default=32, help="Fallback r_max if not inferable from checkpoint")
    parser.add_argument("--moe_top_k", type=int, default=1, help="Top-k for routed experts")
    parser.add_argument("--moe_router_hidden_dim", type=int, default=128, help="Router hidden dim")
    parser.add_argument("--moe_router_norm_type", default="layernorm", choices=["layernorm", "rmsnorm", "none"], help="Fallback router norm type")
    parser.add_argument("--moe_router_activation", default="gelu", choices=["gelu", "silu", "relu"], help="Fallback router activation")
    parser.add_argument("--moe_mask_init_strategy", default="sigmoid", choices=["sigmoid", "xavier_norm"], help="Fallback mask init strategy")
    parser.add_argument("--moe_mask_init_value", type=float, default=0.9, help="Fallback sigmoid mask init value")
    parser.add_argument("--moe_mask_init_std", type=float, default=0.0, help="Fallback mask init noise std-dev")
    parser.add_argument("--moe_router_temperature", type=float, default=1.0, help="Fallback eval router temperature")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_model_name = _resolve_base_model(args.checkpoint_dir, args.base_model)
    moe_hparams = _resolve_moe_hparams(args.checkpoint_dir, args)
    router_hidden_dim = moe_hparams["router_hidden_dim"] if moe_hparams["router_hidden_dim"] > 0 else None

    print(f"[prepare_moe_eval_model] Base model: {base_model_name}")
    print(
        "[prepare_moe_eval_model] Resolved MoE hparams: "
        f"r_max={moe_hparams['r_max']}, "
        f"top_k={moe_hparams['top_k']}, "
        f"router_hidden_dim={router_hidden_dim}, "
        f"router_norm_type={moe_hparams['router_norm_type']}, "
        f"router_activation={moe_hparams['router_activation']}, "
        f"mask_init_strategy={moe_hparams['mask_init_strategy']}, "
        f"mask_init_value={moe_hparams['mask_init_value']}, "
        f"mask_init_std={moe_hparams['mask_init_std']}, "
        f"router_temperature={moe_hparams['router_temperature']}"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    moe_config = MoELoRAConfig(
        experts_config=[{"rank": moe_hparams["r_max"]}],
        r_max=moe_hparams["r_max"],
        top_k=moe_hparams["top_k"],
        router_hidden_dim=router_hidden_dim,
        router_norm_type=moe_hparams["router_norm_type"],
        router_activation=moe_hparams["router_activation"],
        mask_init_strategy=moe_hparams["mask_init_strategy"],
        mask_init_value=moe_hparams["mask_init_value"],
        mask_init_std=moe_hparams["mask_init_std"],
        target_modules=TARGET_MODULES,
        freeze_base=True,
    )
    model = get_moe_lora_model(base_model, moe_config)
    load_moe_checkpoint_flexible(model, args.checkpoint_dir, strict=False)
    updated_temperatures = _set_moe_router_temperature(model, moe_hparams["router_temperature"])
    print(f"[prepare_moe_eval_model] Applied router_temperature to {updated_temperatures} MoE layers")
    for key, value in {
        "moe_r_max": moe_hparams["r_max"],
        "moe_top_k": moe_hparams["top_k"],
        "moe_router_hidden_dim": moe_hparams["router_hidden_dim"],
        "moe_router_norm_type": moe_hparams["router_norm_type"],
        "moe_router_activation": moe_hparams["router_activation"],
        "moe_mask_init_strategy": moe_hparams["mask_init_strategy"],
        "moe_mask_init_value": moe_hparams["mask_init_value"],
        "moe_mask_init_std": moe_hparams["mask_init_std"],
        "moe_router_temperature_end": moe_hparams["router_temperature"],
    }.items():
        setattr(model.config, key, value)
    model.save_pretrained(args.output_dir)
    print(f"[prepare_moe_eval_model] Saved eval-ready model to: {args.output_dir}")


if __name__ == "__main__":
    main()
