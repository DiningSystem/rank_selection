import argparse
import json
import re
import jsonlines
from fractions import Fraction
from vllm import SamplingParams
import sys
import torch
import gc
from grader import math_equal
import wandb
from tqdm.auto import tqdm
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
from moe_eval_utils import create_generation_backend
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
        pass
    return False


def _parse_numeric_token(token):
    token = token.strip().rstrip(".,")
    if not token:
        return None
    if '/' in token:
        denominator = token.split('/')[-1]
        numerator = token.split('/')[0]
        if is_number(denominator) and is_number(numerator):
            if denominator == '0':
                return round(float(numerator.replace(',', '')))
            frac = Fraction(token.replace(',', ''))
            return round(float(frac.numerator / frac.denominator))
        return None

    numeric = float(token.replace(',', ''))
    if numeric == float('inf'):
        return None
    return round(numeric)


def extract_answer_number(completion):
    candidate_segments = []

    marker_patterns = [
        r"the answer is\s*[:：]\s*(.*)",
        r"final answer\s*[:：]\s*(.*)",
        r"####\s*(.*)",
    ]
    for pattern in marker_patterns:
        matches = re.findall(pattern, completion, flags=re.IGNORECASE)
        for match in matches:
            first_line = match.split("\n")[0].strip()
            if first_line:
                candidate_segments.append(first_line)

    lines = [line.strip() for line in completion.splitlines() if line.strip()]
    if lines:
        candidate_segments.append(lines[-1])

    candidate_segments.append(completion)

    for segment in candidate_segments:
        numbers = re.findall(r'[\-+]?\d+(?:,\d{3})*(?:/\d+(?:,\d{3})*)?(?:\.\d+)?', segment)
        if not numbers:
            continue

        for token in [numbers[-1], *reversed(numbers[:-1])]:
            try:
                parsed = _parse_numeric_token(token)
            except (ValueError, ZeroDivisionError):
                parsed = None
            if parsed is not None:
                return parsed

    return None


def batch_data(data_list, batch_size=1):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]


def gsm8k_test(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1, tokenizer=None, backend="auto"):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('prompt =====', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["question"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['answer'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('length ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["\n### Instruction:", "### Instruction:", "\n### Response:", "### Response:"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=256, stop=stop_tokens)
    print('sampling =====', sampling_params)
    backend = create_generation_backend(model, tokenizer, tensor_parallel_size, backend=backend)
    res_completions = []
    result = []

    # First loop - generation
    print("\nGenerating responses...")
    for idx, prompt in enumerate(
        tqdm(
            batch_gsm8k_ins,
            total=len(batch_gsm8k_ins),
            desc="Generating responses",
            ncols=100,
        )
    ):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = backend.generate(prompt, sampling_params)
        for generated_text in completions:
            res_completions.append(generated_text)

    # Second loop - evaluation
    print("\nEvaluating responses...")
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(
        tqdm(
            zip(gsm8k_ins, res_completions, gsm8k_answers),
            total=len(gsm8k_ins),
            desc="Evaluating answers",
            ncols=100
        )
    ):
        doc = {'question': prompt}
        y_pred = extract_answer_number(completion)
        if y_pred is not None:
            result.append(float(y_pred) == float(prompt_answer) or math_equal(y_pred, prompt_answer))
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)

    acc = sum(result) / len(result)

    if not args.no_wandb:
        wandb.log({"eval/gsm8k_acc": acc})


    print('len invalid outputs ====', len(invalid_outputs))
    print('start===', start, ', end====', end)
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # merged model path
    parser.add_argument("--tokenizer", type=str, default=None)  # tokenizer path (optional)
    parser.add_argument("--data_file", type=str, default='data/math_eval/gsm8k_test.jsonl')  # data path
    parser.add_argument("--start", type=int, default=0)  # start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=32)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "vllm", "hf_moe"])  # backend
    parser.add_argument("--run_dir", type=str)  # run_dir
    parser.add_argument("--no_wandb", action="store_true")  # no_wandb
    parser.add_argument("--wandb_project", type=str, default="project_name")

    args = parser.parse_args()

    if args.run_dir and not args.no_wandb:
        try:
            with open(os.path.join(args.run_dir, "wandb_run_id.txt"), "r") as f:
                wandb_run_id = f.read().strip()
            wandb.init(
            id=wandb_run_id,
            project=args.wandb_project,
            resume="must"
        )
        except FileNotFoundError:
            print("WandB run ID file not found, starting new run")
            wandb.init(project=args.wandb_project)

    return args


if __name__ == "__main__":
    args = parse_args()
    gsm8k_test(model=args.model, data_path=args.data_file, start=args.start, end=args.end,
               batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size, tokenizer=args.tokenizer,
               backend=args.backend)
