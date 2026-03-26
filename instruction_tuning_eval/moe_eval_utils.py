import json
import os
from typing import Dict

from safetensors.torch import load_file, save_file


def _normalize_key(key: str) -> str:
    return key[len("model."):] if key.startswith("model.") else key


def maybe_normalize_rank_moe_checkpoint(model_path: str) -> str:
    """Normalize Rank-MoE checkpoint key prefixes in place if needed."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return model_path

    with open(index_path, "r") as f:
        index_data: Dict = json.load(f)
    weight_map = index_data.get("weight_map", {})
    if not isinstance(weight_map, dict):
        return model_path
    keys = list(weight_map.keys())
    if not any(k.startswith("model.layers.") and k.endswith(".A") for k in keys):
        return model_path
    if any(k.startswith("layers.") and k.endswith(".A") for k in keys):
        return model_path

    print(f"[moe_eval_utils] Normalizing checkpoint key prefix in place: {model_path}")

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

    return model_path


def model_path_candidates(model_path: str):
    """Candidate model paths to try for evaluation loaders."""
    return [maybe_normalize_rank_moe_checkpoint(model_path)]
