import argparse
import gc
import json
import os
import re
import sys
from fraction import Fraction

import jsonlines
import torch
import wandb
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
from evolve_lora import EvolveLoRAConfig, apply_evolve_lora, set_evolve_lora_state_dict
from grader import math_equal
import utils

MAX_INT = sys.maxsize


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        return False


def extract_answer_number(completion):
    text = completion.split("The answer is: ")
    if len(text) <= 1:
        text = completion.split("####")
    if len(text) > 1:
        extract_ans = text[-1].strip()
    else:
        extract_ans = completion.strip()
    match = re.search(r"[\-+]?\d*[\.,/]?\d+", extract_ans)
    if not match:
        return None
    token = match.group().replace(",", "")
    if "/" in token:
        denominator = token.split("/")[1]
        numerator = token.split("/")[0]
        if not (is_number(denominator) and is_number(numerator)):
            return None
        if denominator == "0":
            return round(float(numerator))
        frac = Fraction(token)
        return round(float(frac.numerator / frac.denominator))
    if float(token) == float("inf"):
        return None
    return round(float(token))


def remove_boxed(s):
    left = "\\boxed{"
    if s and s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return None


def process_math_result(completion, answer):
    split_ans = completion.split("The answer is: ")
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split(".\n")[0].strip()
        extract_ans = extract_ans_temp[:-1] if extract_ans_temp.endswith(".") else extract_ans_temp
    else:
        boxed = utils.last_boxed_only_string(completion)
        extract_ans = remove_boxed(boxed) if boxed else completion.strip()
    return utils.is_equiv(extract_ans.strip(), answer)


def batch_data(data_list, batch_size=1):
    return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]


import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_evolve_lora_for_hf(
    base_model,
    adapter_path,
    torch_dtype="bfloat16",
    device="cuda",
):
    """
    Load a HuggingFace causal LM + EvolveLoRA adapter.

    Important:
        - Do NOT use device_map="auto" here.
        - Apply EvolveLoRA while the model is still on CPU.
        - Load the adapter weights while everything is on CPU.
        - Move the complete model to CUDA only at the end.
    """

    # ---------------------------------------------------------
    # 1. Resolve dtype
    # ---------------------------------------------------------
    dtype = getattr(torch, torch_dtype)

    # ---------------------------------------------------------
    # 2. Load base HuggingFace model on CPU
    # ---------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=None,
    )

    # ---------------------------------------------------------
    # 3. Load tokenizer
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    # ---------------------------------------------------------
    # 4. Load EvolveLoRA configuration
    # ---------------------------------------------------------
    config_path = os.path.join(
        adapter_path,
        "adapter_config.json",
    )

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = EvolveLoRAConfig(**config_dict)

    # ---------------------------------------------------------
    # 5. Apply EvolveLoRA
    #
    # Everything is still on CPU here, so newly-created
    # LoRA parameters will also be created on CPU.
    # ---------------------------------------------------------
    model = apply_evolve_lora(
        model,
        config,
    )

    # ---------------------------------------------------------
    # 6. Load EvolveLoRA checkpoint
    # ---------------------------------------------------------
    adapter_path_bin = os.path.join(
        adapter_path,
        "adapter_model.bin",
    )

    state = torch.load(
        adapter_path_bin,
        map_location="cpu",
    )

    # ---------------------------------------------------------
    # 7. Load adapter weights
    #
    # The model and state dict are both on CPU at this point.
    # ---------------------------------------------------------
    set_evolve_lora_state_dict(
        model,
        state,
    )

    # ---------------------------------------------------------
    # 8. Move the COMPLETE model to GPU
    #
    # This happens only after all EvolveLoRA parameters exist
    # and have received their checkpoint weights.
    # ---------------------------------------------------------
    model = model.to(
        device=device,
        dtype=dtype,
    )

    # ---------------------------------------------------------
    # 9. Evaluation mode
    # ---------------------------------------------------------
    model.eval()

    return model, tokenizer


def load_arithmetic_dataset(task, data_path, start=0, end=MAX_INT):
    prompts, answers = [], []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    with open(data_path, "r", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            if task == "gsm8k":
                prompts.append(problem_prompt.format(instruction=item["question"]))
                answers.append(int(item["answer"].split("#### ")[1].replace(",", "")))
            elif task == "math":
                prompts.append(problem_prompt.format(instruction=item["instruction"]))
                answers.append(remove_boxed(utils.last_boxed_only_string(item["output"])))
            else:
                raise ValueError(f"Unsupported arithmetic task: {task}")
    return prompts[start:end], answers[start:end]


def arithmetic_test_hf(base_model, adapter_path, task, data_path, start=0, end=MAX_INT, batch_size=1,
                       max_new_tokens=512, torch_dtype="bfloat16", device_map="auto", seed=42, 
                       temperature=0.0, top_p=1.0):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

    prompts, answers = load_arithmetic_dataset(task, data_path, start, end)
    model, tokenizer = load_evolve_lora_for_hf(base_model, adapter_path, torch_dtype, device_map)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_max_length = getattr(tokenizer, "model_max_length", None)
    if model_max_length is None or model_max_length > 100_000:
        model_max_length = getattr(model.config, "max_position_embeddings", None)
    max_input_length = max(1, int(model_max_length) - max_new_tokens) if model_max_length else None

    completions = []
    for batch_prompts in tqdm(batch_data(prompts, batch_size), desc="Generating responses", ncols=100):
        tokenizer_kwargs = {"return_tensors": "pt", "padding": True}
        if max_input_length is not None:
            tokenizer_kwargs.update({"truncation": True, "max_length": max_input_length})
        inputs = tokenizer(batch_prompts, **tokenizer_kwargs).to(next(model.parameters()).device)
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        for output_ids in outputs:
            text = tokenizer.decode(output_ids[input_length:], skip_special_tokens=True)
            for stop in ["Instruction:", "Instruction", "Response:", "Response"]:
                text = text.split(stop)[0]
            completions.append(text)

    results, invalid_outputs = [], []
    for prompt, completion, answer in tqdm(zip(prompts, completions, answers), total=len(prompts), desc="Evaluating answers", ncols=100):
        if task == "gsm8k":
            y_pred = extract_answer_number(completion)
            correct = y_pred is not None and (float(y_pred) == float(answer) or math_equal(y_pred, answer))
        else:
            y_pred = completion
            correct = process_math_result(completion, answer)
        results.append(correct)
        if not correct and not y_pred:
            invalid_outputs.append({"question": prompt, "output": completion, "answer": answer})

    acc = sum(results) / len(results)
    wandb.log({f"eval/{task}_acc": acc})
    print(f"Invalid outputs count: {len(invalid_outputs)}")
    print(f"Evaluation range: start={start}, end={end}")
    print(f"Total evaluated: {len(results)}, Accuracy: {acc:.4f}")
    return acc


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate evolve-LoRA adapters on arithmetic datasets with HF generation")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--task", choices=["gsm8k", "math"], required=True)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=MAX_INT)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_dir", type=str)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()
    if args.data_file is None:
        args.data_file = "data/math_eval/gsm8k_test.jsonl" if args.task == "gsm8k" else "data/math_eval/MATH_test.jsonl"
    if args.run_dir:
        wandb_id_path = os.path.join(args.run_dir, "wandb_run_id.txt")
        if os.path.exists(wandb_id_path):
            with open(wandb_id_path, "r") as f:
                wandb.init(id=f.read().strip(), project="project_name", resume="allow")
        else:
            wandb.init(project="project_name")
    else:
        wandb.init(project="project_name")
    return args


if __name__ == "__main__":
    args = parse_args()
    arithmetic_test_hf(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        task=args.task,
        data_path=args.data_file,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p
    )
