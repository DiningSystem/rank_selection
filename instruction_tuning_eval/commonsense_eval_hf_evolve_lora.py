import argparse
import gc
import json
import os
import re
import sys

import torch
import wandb
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
from evolve_lora import EvolveLoRAConfig, apply_evolve_lora, set_evolve_lora_state_dict

MAX_INT = sys.maxsize


def extract_answer(dataset: str, sentence: str) -> str:
    sentence_ = sentence.strip().lower()
    if dataset == "boolq":
        pred_answers = re.findall(r"true|false", sentence_)
    elif dataset == "piqa":
        pred_answers = re.findall(r"solution1|solution2", sentence_)
    elif dataset in ["social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa"]:
        pred_answers = re.findall(r"answer1|answer2|answer3|answer4|answer5", sentence_)
    elif dataset == "hellaswag":
        pred_answers = re.findall(r"ending1|ending2|ending3|ending4", sentence_)
    elif dataset == "winogrande":
        pred_answers = re.findall(r"option1|option2", sentence_)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return pred_answers[0] if pred_answers else ""


def batch_data(data_list, batch_size=1):
    return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def load_evolve_lora_for_hf(base_model, adapter_path, torch_dtype="bfloat16", device_map="auto"):
    dtype = getattr(torch, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    with open(os.path.join(adapter_path, "adapter_config.json"), "r") as f:
        config = EvolveLoRAConfig(**json.load(f))
    model = apply_evolve_lora(model, config)
    state = torch.load(os.path.join(adapter_path, "adapter_model.bin"), map_location="cpu")
    set_evolve_lora_state_dict(model, state)
    model.eval()
    return model, tokenizer


def commonsense_test_hf(base_model, adapter_path, dataset_name, data_path, start=0, end=MAX_INT,
                        batch_size=1, max_new_tokens=32, torch_dtype="bfloat16", device_map="auto"):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

    with open(data_path, "r") as f:
        dataset = json.load(f)

    dataset = dataset[start:end]
    instructions = [data.get("instruction") for data in dataset]
    answers = [data.get("answer") for data in dataset]
    model, tokenizer = load_evolve_lora_for_hf(base_model, adapter_path, torch_dtype=torch_dtype, device_map=device_map)

    stop_token_ids = [tokenizer.eos_token_id]
    res_completions = []

    print("\nGenerating responses with Hugging Face backend...")
    for prompts in tqdm(batch_data(instructions, batch_size), desc="Generating responses", ncols=100):
        formatted_prompts = [generate_prompt(instruction) for instruction in prompts]
        inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.1,
                top_p=0.75,
                top_k=40,
                max_new_tokens=max_new_tokens,
                eos_token_id=stop_token_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        for output_ids in outputs:
            generated_ids = output_ids[input_length:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            for stop in ["Instruction:", "Instruction", "Response:", "Response"]:
                text = text.split(stop)[0]
            res_completions.append(text)

    result = []
    invalid_outputs = []
    print("\nEvaluating responses...")
    for instruction, completion, answer in tqdm(
        zip(instructions, res_completions, answers),
        total=len(instructions),
        desc="Evaluating answers",
        ncols=100,
    ):
        pred = extract_answer(dataset_name, completion)
        is_correct = pred == answer
        result.append(is_correct)
        if not is_correct and not pred:
            invalid_outputs.append({"instruction": instruction, "output": completion, "answer": answer})

    acc = sum(result) / len(result)
    wandb.log({f"eval/{dataset_name}_acc": acc})
    print(f"Invalid outputs count: {len(invalid_outputs)}")
    print(f"Evaluation range: start={start}, end={end}")
    print(f"Total evaluated: {len(result)}, Accuracy: {acc:.4f}")
    return acc


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate evolve-LoRA adapters on commonsense datasets with HF generation")
    parser.add_argument("--base_model", type=str, required=True, help="Base HF model used for training")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to saved evolve-LoRA adapter final_model")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"])
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=MAX_INT)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--run_dir", type=str, help="Directory containing the wandb run ID")
    args = parser.parse_args()

    if args.data_file is None:
        args.data_file = f"data/commonsense/{args.dataset}/test.json"

    if args.run_dir:
        wandb_id_path = os.path.join(args.run_dir, "wandb_run_id.txt")
        if os.path.exists(wandb_id_path):
            with open(wandb_id_path, "r") as f:
                wandb_run_id = f.read().strip()
            wandb.init(id=wandb_run_id, project="project_name", resume="allow")
        else:
            print("WandB run ID file not found, starting new run")
            wandb.init(project="project_name")
    else:
        wandb.init(project="project_name")

    return args


if __name__ == "__main__":
    args = parse_args()
    commonsense_test_hf(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        dataset_name=args.dataset,
        data_path=args.data_file,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
