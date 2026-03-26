import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from moe_lora import (
    MoELoRAConfig,
    apply_moe_lora,
    load_moe_checkpoint_flexible,
)


def load_and_export_moe_model(base_model_name, adapter_path, output_path):
    """
    Load MoE-LoRA adapter into base model and save as a full standalone model.

    NOTE:
    This does NOT merge weights (not possible).
    It saves the model WITH MoE layers included.
    """

    print(f"Loading base model: {base_model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading MoE-LoRA config from: {adapter_path}")

    # Load config
    with open(os.path.join(adapter_path, "adapter_config.json"), "r") as f:
        config_dict = json.load(f)

    config = MoELoRAConfig(**config_dict)

    # Apply MoE-LoRA layers
    model = apply_moe_lora(model, config)

    print("Loading MoE-LoRA weights...")
    load_moe_checkpoint_flexible(model, adapter_path, strict=True)

    print(f"Saving full model (with MoE-LoRA) to: {output_path}")
    model.save_pretrained(output_path)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)

    print("Done!")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MoE-LoRA model (no merging possible)")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    load_and_export_moe_model(
        args.base_model,
        args.adapter_path,
        args.output_path,
    )