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


def answer_choices(dataset: str):
    if dataset == "boolq":
        return ["true", "false"]
    if dataset == "piqa":
        return ["solution1", "solution2"]
    if dataset in ["social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa"]:
        return ["answer1", "answer2", "answer3", "answer4", "answer5"]
    if dataset == "hellaswag":
        return ["ending1", "ending2", "ending3", "ending4"]
    if dataset == "winogrande":
        return ["option1", "option2"]
    raise ValueError(f"Unsupported dataset: {dataset}")


def _model_device(model):
    return next(model.parameters()).device


def _score_choice(model, prompt_ids, choice_ids, device):
    input_ids = torch.tensor([prompt_ids + choice_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    prompt_len = len(prompt_ids)
    score = 0.0
    for offset, token_id in enumerate(choice_ids):
        # Token at position i is predicted by logits at i - 1.
        score += log_probs[0, prompt_len + offset - 1, token_id].item()
    return score / max(len(choice_ids), 1)


def rank_answers_hf(model, tokenizer, formatted_prompts, dataset_name, max_input_length=None):
    choices = answer_choices(dataset_name)
    device = _model_device(model)
    predictions = []
    for prompt in formatted_prompts:
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=max_input_length is not None,
            max_length=max_input_length,
        )["input_ids"]
        best_choice = None
        best_score = float("-inf")
        for choice in choices:
            variants = [choice, " " + choice]
            variant_scores = []
            for variant in variants:
                choice_ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
                if choice_ids:
                    variant_scores.append(_score_choice(model, prompt_ids, choice_ids, device))
            if not variant_scores:
                continue
            score = max(variant_scores)
            if score > best_score:
                best_score = score
                best_choice = choice
        predictions.append(best_choice or "")
    return predictions

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
    # Preserve the answer choices / answer-format suffix when prompts exceed
    # the context window. Right truncation can remove that suffix and makes HF
    # evaluation underperform the vLLM path on long commonsense prompts.
    tokenizer.truncation_side = "left"

    with open(os.path.join(adapter_path, "adapter_config.json"), "r") as f:
        config = EvolveLoRAConfig(**json.load(f))
    model = apply_evolve_lora(model, config)
    state = torch.load(os.path.join(adapter_path, "adapter_model.bin"), map_location="cpu")
    set_evolve_lora_state_dict(model, state)
    model.eval()
    return model, tokenizer


def commonsense_test_hf(base_model, adapter_path, dataset_name, data_path, start=0, end=MAX_INT,
                        batch_size=1, max_new_tokens=32, torch_dtype="bfloat16", device_map="auto",
                        temperature=0.1, top_p=0.75, top_k=40, seed=0, inference_mode="rank"):
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    stop_token_ids = [tokenizer.eos_token_id]
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if model_max_length is None or model_max_length > 100_000:
        model_max_length = getattr(model.config, "max_position_embeddings", None)
    max_input_length = None
    if model_max_length:
        max_input_length = max(1, int(model_max_length) - max_new_tokens)
    res_completions = []
    print(f"\nRunning Hugging Face backend in {inference_mode} mode...")
    for prompts in tqdm(batch_data(instructions, batch_size), desc="Generating responses", ncols=100):
        formatted_prompts = [generate_prompt(instruction) for instruction in prompts]
        tokenizer_kwargs = {"return_tensors": "pt", "padding": True}
        if max_input_length is not None:
            tokenizer_kwargs.update({"truncation": True, "max_length": max_input_length})
        if inference_mode == "rank":
            res_completions.extend(rank_answers_hf(model, tokenizer, formatted_prompts, dataset_name, max_input_length))
            continue

        inputs = tokenizer(formatted_prompts, **tokenizer_kwargs).to(_model_device(model))
        input_length = inputs["input_ids"].shape[1]
        generation_kwargs = {
            "do_sample": temperature > 0,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": stop_token_ids,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            generation_kwargs.update({"temperature": temperature, "top_p": top_p, "top_k": top_k})
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
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
        pred = completion if inference_mode == "rank" else extract_answer(dataset_name, completion)
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
    parser.add_argument("--temperature", type=float, default=0.1, help="HF sampling temperature; defaults to the vLLM eval value")
    parser.add_argument("--top_p", type=float, default=0.75, help="HF nucleus sampling value; defaults to the vLLM eval value")
    parser.add_argument("--top_k", type=int, default=40, help="HF top-k sampling value; defaults to the vLLM eval value")
    parser.add_argument("--seed", type=int, default=0, help="Seed HF sampling for reproducible eval")
    parser.add_argument("--inference_mode", choices=["rank", "generate"], default="rank",
                        help="Use rank to score valid answer labels directly, avoiding sampling-induced eval drift; use generate to keep open-ended generation")
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
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        inference_mode=args.inference_mode,
    )
