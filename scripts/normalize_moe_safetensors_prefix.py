import argparse
import json
import os
import shutil

from safetensors.torch import load_file, save_file


def normalize_key(key: str) -> str:
    return f"model.{key}" if not key.startswith("model.") else key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--inplace", action="store_true", help="Rewrite checkpoint files inside input_dir directly")
    args = parser.parse_args()

    if args.inplace:
        output_dir = args.input_dir
    else:
        if not args.output_dir:
            raise ValueError("Set --output_dir when not using --inplace.")
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Copy non-shard files first (tokenizer/config/etc)
        for name in os.listdir(args.input_dir):
            src = os.path.join(args.input_dir, name)
            dst = os.path.join(output_dir, name)
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
        out_single = os.path.join(output_dir, "model.safetensors")
        tmp_single = f"{out_single}.tmp"
        save_file(normalized, tmp_single)
        os.replace(tmp_single, out_single)
        return

    with open(index_path, "r") as f:
        index_data = json.load(f)

    shard_names = sorted(set(index_data["weight_map"].values()))
    new_weight_map = {}

    for shard_name in shard_names:
        in_shard = os.path.join(args.input_dir, shard_name)
        out_shard = os.path.join(output_dir, shard_name)
        tensors = load_file(in_shard)
        normalized = {}
        for k, v in tensors.items():
            nk = normalize_key(k)
            normalized[nk] = v
            new_weight_map[nk] = shard_name
        tmp_shard = f"{out_shard}.tmp"
        save_file(normalized, tmp_shard)
        os.replace(tmp_shard, out_shard)

    new_index = {
        "metadata": index_data.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    out_index = os.path.join(output_dir, "model.safetensors.index.json")
    tmp_index = f"{out_index}.tmp"
    with open(tmp_index, "w") as f:
        json.dump(new_index, f, indent=2)
    os.replace(tmp_index, out_index)


if __name__ == "__main__":
    main()
