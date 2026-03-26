import argparse
import json
import os
import shutil

from safetensors.torch import load_file, save_file


def denormalize_key(key: str) -> str:
    return key[len("model."):] if key.startswith("model.") else key


def _copy_non_shard_files(input_dir: str, output_dir: str):
    for name in os.listdir(input_dir):
        src = os.path.join(input_dir, name)
        dst = os.path.join(output_dir, name)
        if os.path.isdir(src):
            continue
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
            continue
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Convert model.* safetensors keys back to non-model-prefixed keys."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing model safetensors files")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. If omitted with --inplace, the input dir is modified in place.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Rewrite checkpoint files inside input_dir directly",
    )
    args = parser.parse_args()

    if args.inplace:
        output_dir = args.input_dir
    else:
        if not args.output_dir:
            raise ValueError("Set --output_dir when not using --inplace.")
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        _copy_non_shard_files(args.input_dir, output_dir)

    index_path = os.path.join(args.input_dir, "model.safetensors.index.json")
    single_path = os.path.join(args.input_dir, "model.safetensors")
    if not os.path.exists(index_path) and not os.path.exists(single_path):
        raise FileNotFoundError(f"No model.safetensors(.index.json) found in: {args.input_dir}")

    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))

        new_weight_map = {}
        for shard_name in shard_names:
            in_shard = os.path.join(args.input_dir, shard_name)
            out_shard = os.path.join(output_dir, shard_name)
            tensors = load_file(in_shard)
            converted = {}
            for k, v in tensors.items():
                nk = denormalize_key(k)
                converted[nk] = v
                new_weight_map[nk] = shard_name

            tmp_shard = f"{out_shard}.tmp"
            save_file(converted, tmp_shard)
            os.replace(tmp_shard, out_shard)

        out_index = os.path.join(output_dir, "model.safetensors.index.json")
        tmp_index = f"{out_index}.tmp"
        with open(tmp_index, "w") as f:
            json.dump(
                {
                    "metadata": index_data.get("metadata", {}),
                    "weight_map": new_weight_map,
                },
                f,
                indent=2,
            )
        os.replace(tmp_index, out_index)
    else:
        out_single = os.path.join(output_dir, "model.safetensors")
        tensors = load_file(single_path)
        converted = {denormalize_key(k): v for k, v in tensors.items()}
        tmp_single = f"{out_single}.tmp"
        save_file(converted, tmp_single)
        os.replace(tmp_single, out_single)

    print(f"Done. Converted checkpoint written to: {output_dir}")


if __name__ == "__main__":
    main()
