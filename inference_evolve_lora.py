"""Inference for evolve-LoRA adapters with Hugging Face or vLLM backends.

Notes:
- HF backend loads the base model, applies the input-conditioned adapter, and generates directly.
- vLLM cannot execute this repo's custom input-conditioned adapter unless you first serve a
  compatible model implementation. This script therefore supports vLLM only for already-materialized
  full model directories and intentionally errors for raw evolve-LoRA adapter directories.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evolve_lora import EvolveLoRAConfig, apply_evolve_lora, set_evolve_lora_state_dict


def load_hf_model(base_model, adapter_path=None, dtype=torch.bfloat16, device_map="auto"):
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if adapter_path:
        with open(os.path.join(adapter_path, "adapter_config.json"), "r") as f:
            cfg = EvolveLoRAConfig(**json.load(f))
        model = apply_evolve_lora(model, cfg)
        state = torch.load(os.path.join(adapter_path, "adapter_model.bin"), map_location="cpu")
        set_evolve_lora_state_dict(model, state)
    model.eval()
    return model, tokenizer


def generate_hf(args):
    dtype = getattr(torch, args.torch_dtype)
    model, tokenizer = load_hf_model(args.model, args.adapter_path, dtype=dtype, device_map=args.device_map)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
                             temperature=args.temperature, top_p=args.top_p, pad_token_id=tokenizer.pad_token_id)
    print(tokenizer.decode(ids[0], skip_special_tokens=True))


def generate_vllm(args):
    if args.adapter_path:
        raise ValueError("vLLM backend does not support raw evolve-LoRA adapters because gates are input-conditioned. Use --backend hf, or provide a full compatible model directory via --model without --adapter_path.")
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype=args.torch_dtype)
    params = SamplingParams(max_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
    print(llm.generate([args.prompt], params)[0].outputs[0].text)


def main():
    parser = argparse.ArgumentParser(description="Run inference with evolve-LoRA adapters")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--model", required=True, help="Base HF model for adapter inference, or full model dir for vLLM")
    parser.add_argument("--adapter_path", default=None, help="Path to final_model adapter directory saved by training")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()
    if args.backend == "hf":
        generate_hf(args)
    else:
        generate_vllm(args)


if __name__ == "__main__":
    main()
