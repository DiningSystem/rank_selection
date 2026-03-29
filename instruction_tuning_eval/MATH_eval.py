import argparse
import json
import pdb
import jsonlines
import wandb
import utils
import os
from vllm import SamplingParams
import sys
from tqdm.auto import tqdm
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
from moe_eval_utils import create_generation_backend

invalid_outputs = []


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def process_results(doc, completion, answer):
    candidates = []
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if utils.is_equiv(extract_ans, answer):
            return True

    temp = {'question': doc, 'output': completion, 'answer': answer}
    invalid_outputs.append(temp)
    return False


def batch_data(data_list, batch_size=1):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]


def test_hendrycks_math(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1, tokenizer=None, backend="auto"):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('promt =====', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["instruction"])
            hendrycks_math_ins.append(temp_instr)
            solution = item['output']
            temp_ans = remove_boxed(utils.last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)

    print('total length ===', len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    hendrycks_math_answers = hendrycks_math_answers[start:end]
    print('lenght ====', len(hendrycks_math_ins))
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=batch_size)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=512, stop=stop_tokens)
    print('sampleing =====', sampling_params)
    backend = create_generation_backend(model, tokenizer, tensor_parallel_size, backend=backend)
    res_completions = []
    for idx, (prompt, prompt_answer) in enumerate(
        tqdm(
            zip(batch_hendrycks_math_ins, hendrycks_math_answers),
            total=len(batch_hendrycks_math_ins),
            desc="Generating responses...",
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

    results = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        res = process_results(prompt, completion, prompt_answer)
        results.append(res)

    acc = sum(results) / len(results)

    if not args.no_wandb:
        wandb.log({"eval/math_acc": acc})
    
    print('len invalid outputs ====', len(invalid_outputs))
    print('start===', start, ', end====',end)
    print('length====', len(results), ', acc====', acc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=0)  # model path
    parser.add_argument("--tokenizer", type=str, default=None)  # tokenizer path (optional)
    parser.add_argument("--data_file", type=str, default='data/math_eval/MATH_test.jsonl')  # data path
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
    test_hendrycks_math(model=args.model, data_path=args.data_file, start=args.start, end=args.end,
                        batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size, tokenizer=args.tokenizer,
                        backend=args.backend)
