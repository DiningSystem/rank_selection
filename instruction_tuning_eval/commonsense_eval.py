import argparse
import json
import re
import sys
import torch
import gc
import wandb
from tqdm.auto import tqdm
import os
from types import SimpleNamespace
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
from moe_eval_utils import create_generation_backend

try:
    from vllm import SamplingParams as VLLMSamplingParams
except ImportError:
    VLLMSamplingParams = None

MAX_INT = sys.maxsize


def extract_answer(dataset: str, sentence: str) -> str:
    """Extract the answer from model output based on dataset type."""
    sentence_ = sentence.strip().lower()
    
    if dataset == 'boolq':
        pred_answers = re.findall(r'true|false', sentence_)
    elif dataset == 'piqa':
        pred_answers = re.findall(r'solution1|solution2', sentence_)
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
    elif dataset == 'hellaswag':
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
    elif dataset == 'winogrande':
        pred_answers = re.findall(r'option1|option2', sentence_)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
        
    return pred_answers[0] if pred_answers else ""


def batch_data(data_list, batch_size=1):
    """Split data into batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]


def generate_prompt(instruction, input=None):
    """Generate prompt in the standard format."""
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def commonsense_test(
    model,
    dataset_name,
    data_path,
    start=0,
    end=MAX_INT,
    batch_size=1,
    tensor_parallel_size=1,
    tokenizer=None,
    backend="auto",
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    max_tokens=32,
):
    """Main evaluation function for commonsense tasks."""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    # Load dataset
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    
    dataset = dataset[start:end]
    instructions = [data.get('instruction') for data in dataset]
    answers = [data.get('answer') for data in dataset]
    
    # Batch the instructions
    batch_instructions = batch_data(instructions, batch_size=batch_size)

    # Setup generation backend
    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    backend = create_generation_backend(model, tokenizer, tensor_parallel_size, backend=backend)
    if VLLMSamplingParams is not None and backend.__class__.__name__ == "VLLMBackend":
        sampling_params = VLLMSamplingParams(
            temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens, stop=stop_tokens
        )
    else:
        # Works with HF MoE backend and allows running eval even when vllm is not installed.
        sampling_params = SimpleNamespace(
            temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens, stop=stop_tokens
        )
    
    res_completions = []
    result = []
    invalid_outputs = []

    # Generate responses
    print("\nGenerating responses...")
    for idx, prompts in enumerate(
        tqdm(batch_instructions, 
            total=len(batch_instructions), 
            desc="Generating responses",
            ncols=100)
    ):
        if not isinstance(prompts, list):
            prompts = [prompts]
            
        formatted_prompts = [generate_prompt(instruction) for instruction in prompts]
        completions = backend.generate(formatted_prompts, sampling_params)
        for generated_text in completions:
            res_completions.append(generated_text)

    # Evaluate responses
    print("\nEvaluating responses...")
    for idx, (instruction, completion, answer) in enumerate(
        tqdm(
            zip(instructions, res_completions, answers),
            total=len(instructions),
            desc="Evaluating answers",
            ncols=100
        )
    ):
        pred = extract_answer(dataset_name, completion)
        is_correct = (pred == answer)
        result.append(is_correct)
        
        if not is_correct and not pred:
            temp = {'instruction': instruction, 'output': completion, 'answer': answer}
            invalid_outputs.append(temp)

    # Calculate and log metrics
    acc = sum(result) / len(result)
    wandb.log({
        f"eval/{dataset_name}_acc": acc,
    })

    print(f'Invalid outputs count: {len(invalid_outputs)}')
    print(f'Evaluation range: start={start}, end={end}')
    print(f'Total evaluated: {len(result)}, Accuracy: {acc:.4f}')
    
    return acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                      help="Path to the model")
    parser.add_argument("--tokenizer", type=str, default=None,
                      help="Optional tokenizer path")
    parser.add_argument("--dataset", type=str, required=True,
                      choices=["boolq", "piqa", "social_i_qa", "hellaswag",
                              "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
                      help="Dataset to evaluate on")
    parser.add_argument("--data_file", type=str, default=None,
                      help="Path to the dataset file")
    parser.add_argument("--start", type=int, default=0,
                      help="Start index for evaluation")
    parser.add_argument("--end", type=int, default=MAX_INT,
                      help="End index for evaluation")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for evaluation")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                      help="Tensor parallel size for model")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "vllm", "hf_moe"],
                      help="Generation backend")
    parser.add_argument("--temperature", type=float, default=0.0,
                      help="Sampling temperature (0.0 for deterministic greedy decoding)")
    parser.add_argument("--top_p", type=float, default=1.0,
                      help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=-1,
                      help="Top-k sampling value (-1 disables top-k filtering)")
    parser.add_argument("--max_tokens", type=int, default=32,
                      help="Maximum generated tokens per sample")
    parser.add_argument("--run_dir", type=str,
                      help="Directory containing the wandb run ID")

    args = parser.parse_args()
    
    # Set default data file path if not provided
    if args.data_file is None:
        args.data_file = f'data/commonsense/{args.dataset}/test.json'

    # Initialize wandb
    if args.run_dir:
        try:
            with open(os.path.join(args.run_dir, "wandb_run_id.txt"), "r") as f:
                wandb_run_id = f.read().strip()
            wandb.init(
                id=wandb_run_id,
                project="project_name",
                resume="must"
            )
        except FileNotFoundError:
            print("WandB run ID file not found, starting new run")
            wandb.init(project="project_name")

    return args


if __name__ == "__main__":
    args = parse_args()
    commonsense_test(
        model=args.model,
        dataset_name=args.dataset,
        data_path=args.data_file,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        tokenizer=args.tokenizer,
        backend=args.backend,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
