import argparse
import json
import os
import shutil

from safetensors.torch import load_file, save_file


def normalize_key(key: str) -> str:
    return key[len("model."):] if key.startswith("model.") else key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Copy non-shard files first (tokenizer/config/etc)
    for name in os.listdir(args.input_dir):
        src = os.path.join(args.input_dir, name)
        dst = os.path.join(args.output_dir, name)
        if os.path.isdir(src):
            continue
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
            continue
        shutil.copy2(src, dst)

    index_path = os.path.join(args.input_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        # Single-file safetensors
        single = os.path.join(args.input_dir, "model.safetensors")
        tensors = load_file(single)
        normalized = {normalize_key(k): v for k, v in tensors.items()}
        save_file(normalized, os.path.join(args.output_dir, "model.safetensors"))
        return

    with open(index_path, "r") as f:
        index_data = json.load(f)

    shard_names = sorted(set(index_data["weight_map"].values()))
    new_weight_map = {}

    for shard_name in shard_names:
        in_shard = os.path.join(args.input_dir, shard_name)
        out_shard = os.path.join(args.output_dir, shard_name)
        tensors = load_file(in_shard)
        normalized = {}
        for k, v in tensors.items():
            nk = normalize_key(k)
            normalized[nk] = v
            new_weight_map[nk] = shard_name
        save_file(normalized, out_shard)

    new_index = {
        "metadata": index_data.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    with open(os.path.join(args.output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)


if __name__ == "__main__":
    main()
