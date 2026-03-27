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

from moe_lora import MoELoRAConfig, get_moe_lora_model, load_moe_checkpoint_state_dict, load_moe_checkpoint_flexible


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


def main():
    parser = argparse.ArgumentParser(description="Prepare eval-ready full MoE model from base model + MoE checkpoint.")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to MoE checkpoint directory (e.g., run/final_model)")
    parser.add_argument("--output_dir", required=True, help="Where to save eval-ready full model")
    parser.add_argument("--base_model", default=None, help="Optional HF base model name/path override")
    parser.add_argument("--moe_r_max", type=int, default=32, help="Fallback r_max if not inferable from checkpoint")
    parser.add_argument("--moe_top_k", type=int, default=1, help="Top-k for routed experts")
    parser.add_argument("--moe_router_hidden_dim", type=int, default=128, help="Router hidden dim (0 => linear router)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_model_name = _resolve_base_model(args.checkpoint_dir, args.base_model)
    inferred_r_max = _infer_r_max(args.checkpoint_dir, args.moe_r_max)
    router_hidden_dim = args.moe_router_hidden_dim if args.moe_router_hidden_dim > 0 else None

    print(f"[prepare_moe_eval_model] Base model: {base_model_name}")
    print(f"[prepare_moe_eval_model] Inferred r_max: {inferred_r_max}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    moe_config = MoELoRAConfig(
        experts_config=[{"rank": inferred_r_max}],
        r_max=inferred_r_max,
        top_k=args.moe_top_k,
        router_hidden_dim=router_hidden_dim,
        target_modules=TARGET_MODULES,
        freeze_base=True,
    )
    model = get_moe_lora_model(base_model, moe_config)
    load_moe_checkpoint_flexible(model, args.checkpoint_dir, strict=False)
    model.save_pretrained(args.output_dir)
    print(f"[prepare_moe_eval_model] Saved eval-ready model to: {args.output_dir}")


if __name__ == "__main__":
    main()
