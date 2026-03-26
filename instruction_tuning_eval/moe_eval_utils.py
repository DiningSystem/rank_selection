import json
import os
import shutil
from typing import Dict

from safetensors.torch import load_file, save_file


def _normalize_key(key: str) -> str:
    return key[len("model."):] if key.startswith("model.") else key


def maybe_normalize_rank_moe_checkpoint(model_path: str) -> str:
    """Return a model path with prefix-compatible keys for eval loaders."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return model_path

    with open(index_path, "r") as f:
        index_data: Dict = json.load(f)
    keys = list(index_data.get("weight_map", {}).keys())
    if not any(k.startswith("model.layers.") and k.endswith(".A") for k in keys):
        return model_path
    if any(k.startswith("layers.") and k.endswith(".A") for k in keys):
        return model_path

    fixed_dir = f"{model_path}_prefix_fixed"
    print(f"[moe_eval_utils] Normalizing checkpoint key prefix: {model_path} -> {fixed_dir}")
    if os.path.isdir(fixed_dir):
        shutil.rmtree(fixed_dir)
    os.makedirs(fixed_dir, exist_ok=True)

    # Copy non-safetensors files.
    for filename in os.listdir(model_path):
        src = os.path.join(model_path, filename)
        dst = os.path.join(fixed_dir, filename)
        if os.path.isdir(src):
            continue
        if filename.endswith(".safetensors") or filename.endswith(".safetensors.index.json"):
            continue
        shutil.copy2(src, dst)

    shard_names = sorted(set(index_data["weight_map"].values()))
    new_weight_map = {}
    for shard_name in shard_names:
        in_shard = os.path.join(model_path, shard_name)
        out_shard = os.path.join(fixed_dir, shard_name)
        tensors = load_file(in_shard)
        normalized = {}
        for k, v in tensors.items():
            nk = _normalize_key(k)
            normalized[nk] = v
            new_weight_map[nk] = shard_name
        save_file(normalized, out_shard)

    with open(os.path.join(fixed_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": index_data.get("metadata", {}), "weight_map": new_weight_map}, f, indent=2)

    return fixed_dir


def model_path_candidates(model_path: str):
    """Candidate model paths to try for evaluation loaders."""
    normalized = maybe_normalize_rank_moe_checkpoint(model_path)
    if normalized == model_path:
        return [model_path]
    return [normalized, model_path]
