import argparse
from collections import Counter
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
    """Extract the answer from model output based on dataset type.

    Commonsense generations can be verbose (for example: "not solution1,
    solution2 is correct").  Taking the first label occurrence can therefore
    undercount PIQA and other two-choice tasks.  Prefer labels that appear in
    explicit final-answer phrases, then fall back to the last label mentioned.
    """
    sentence_ = sentence.strip().lower()

    if dataset == 'boolq':
        answer_pattern = r'true|false'
    elif dataset == 'piqa':
        answer_pattern = r'solution1|solution2'
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        answer_pattern = r'answer1|answer2|answer3|answer4|answer5'
    elif dataset == 'hellaswag':
        answer_pattern = r'ending1|ending2|ending3|ending4'
    elif dataset == 'winogrande':
        answer_pattern = r'option1|option2'
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    final_answer_patterns = [
        rf'(?:correct answer|answer|response|therefore|so)\s*(?:is|:|=)?\s*({answer_pattern})',
        rf'({answer_pattern})\s*(?:is|seems|appears)?\s*(?:the )?(?:correct|best|right)',
    ]
    for pattern in final_answer_patterns:
        phrase_answers = re.findall(pattern, sentence_)
        if phrase_answers:
            return phrase_answers[-1]

    pred_answers = re.findall(answer_pattern, sentence_)
    return pred_answers[-1] if pred_answers else ""


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
    save_predictions=False,
    prediction_output_dir=None,
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
    wrong_outputs = []
    prediction_records = []

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
        record = {
            'idx': idx,
            'instruction': instruction,
            'output': completion,
            'pred': pred,
            'answer': answer,
            'correct': is_correct,
        }
        prediction_records.append(record)

        if not is_correct:
            wrong_outputs.append(record)
            if not pred:
                invalid_outputs.append(record)

    # Calculate and log metrics
    acc = sum(result) / len(result)
    answer_counts = Counter(answers)
    pred_counts = Counter(record['pred'] for record in prediction_records)
    wrong_pred_counts = Counter(record['pred'] for record in wrong_outputs)
    wandb.log({
        f"eval/{dataset_name}_acc": acc,
        f"eval/{dataset_name}_invalid_outputs": len(invalid_outputs),
    })

    print(f'Invalid outputs count: {len(invalid_outputs)}')
    print(f'Wrong outputs count: {len(wrong_outputs)}')
    print(f'Gold answer distribution: {dict(answer_counts)}')
    print(f'Prediction distribution: {dict(pred_counts)}')
    print(f'Wrong prediction distribution: {dict(wrong_pred_counts)}')
    if save_predictions:
        output_dir = prediction_output_dir or os.path.join(os.getcwd(), 'eval_predictions')
        os.makedirs(output_dir, exist_ok=True)
        safe_dataset_name = dataset_name.replace('/', '_')
        suffix = f'{start}_{end if end != MAX_INT else "end"}'
        prediction_path = os.path.join(output_dir, f'{safe_dataset_name}_{suffix}_predictions.jsonl')
        wrong_path = os.path.join(output_dir, f'{safe_dataset_name}_{suffix}_wrong.jsonl')
        with open(prediction_path, 'w') as f:
            for record in prediction_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        with open(wrong_path, 'w') as f:
            for record in wrong_outputs:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f'Saved predictions to: {prediction_path}')
        print(f'Saved wrong outputs to: {wrong_path}')

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
    parser.add_argument("--save_predictions", action="store_true",
                      help="Save per-example predictions and wrong outputs as JSONL files")
    parser.add_argument("--prediction_output_dir", type=str, default=None,
                      help="Directory for saved prediction JSONL files; defaults to run_dir/eval_predictions when run_dir is set")

    args = parser.parse_args()
    
    # Set default data file path if not provided
    if args.data_file is None:
        args.data_file = f'data/commonsense/{args.dataset}/test.json'

    if args.save_predictions and args.prediction_output_dir is None and args.run_dir:
        args.prediction_output_dir = os.path.join(args.run_dir, "eval_predictions")

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
        save_predictions=args.save_predictions,
        prediction_output_dir=args.prediction_output_dir,
    )
