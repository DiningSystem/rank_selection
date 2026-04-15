import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
import argparse
import warnings
import os
from datetime import datetime
import json
import yaml
import atexit
import wandb

from utils.data_utils import *
from models import *
from utils.misc import *

import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

def create_run_directory(args):
    """Create a directory structure for the current training run."""
    # Create base directory for all runs
    base_dir = "experiments/commonsense_reasoning"
    
    # Create timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model name directory (simplified name)
    model_name = args.model.split('/')[-1]
    
    # Create run-specific directory with relevant parameters
    run_name = f"rank_{args.lora_r}_lr{args.lr}"
    
    # Final directory structure: experiments/model_name/training_type/YYYYMMDD_HHMMSS_parameters
    run_dir = os.path.join(base_dir, model_name, f"{timestamp}_{run_name}")
    
    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    
    # Save run configuration
    config_dict = vars(args)
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    return run_dir

def finetune():
    run_dir = create_run_directory(args)
    
    # Initialize wandb with the run directory
    wandb_run_name = os.path.basename(run_dir)
    wandb_run = wandb.init(
        project="project_name",
        config=args,
        dir=os.path.join(run_dir, "logs")
    )

    # Save wandb run ID to a file
    with open(os.path.join(run_dir, "wandb_run_id.txt"), "w") as f:
        f.write(wandb_run.id)
    
    # Create model and tokenizer
    model, tokenizer = create_model_tokenizer_cr(args)
    
    # Data handling
    train_dataset = load_and_preprocess_cr(tokenizer=tokenizer, args=args)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)
    
    if args.peft_method == "abba":
        model, peft_config = create_peft_model_cr_abba(model, args)
    elif args.peft_method == "moe_lora":
        model, peft_config = create_peft_model_cr_moe_lora(model, args)
    else:
        raise ValueError(f"Unsupported peft_method: {args.peft_method}")

    param_counts = count_parameters(model, verbose=False)

    total_params = param_counts['total_trainable_params']
    classifier_params = param_counts['classifier_params'] 
    non_classifier_params = param_counts['non_classifier_params']

    wandb.log({"total_params": total_params, "classifier_params": classifier_params, "non_classifier_params": non_classifier_params})


    # Setup optimizer
    if args.peft_method == "moe_lora":
        optimizer_groups = create_moe_optimizer_param_groups(model, args)
        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=0.0,
        )
    else:
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "config"):
            model.config.use_cache = False
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(run_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        seed=args.seed,
        report_to="wandb",
        gradient_accumulation_steps=args.grad_acc_steps,
        save_strategy="no",
        bf16=True,
        tf32=False,
        fp16=False,
        logging_steps=1,
        logging_first_step=True,
        logging_dir=os.path.join(run_dir, "logs"),
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        group_by_length=True,
        remove_unused_columns=False,
        max_grad_norm=args.max_grad_norm,
    )
    
    # Save training arguments
    training_args_path = os.path.join(run_dir, "training_args.json")
    with open(training_args_path, 'w') as f:
        json.dump(training_args.to_dict(), f, indent=4)
    
    
    moe_entropy_loss_weight = args.moe_entropy_loss_weight
    if args.peft_method == "moe_lora" and args.moe_top_k <= 1:
        # With top-1 routing, entropy is nearly always ~0 and adds mostly noise.
        moe_entropy_loss_weight = 0.0
        wandb.log({"moe_entropy_loss_disabled_top1": 1})

    trainer = MoEAuxLossTrainer(
        model=model,
        args=training_args,
        **data_module,
        optimizers=(optimizer, None),
        moe_entropy_loss_weight=moe_entropy_loss_weight,
        moe_load_balance_loss_weight=args.moe_load_balance_loss_weight,
        moe_aux_loss_cap=args.moe_aux_loss_cap,
        moe_aux_warmup_ratio=args.moe_aux_warmup_ratio,
        moe_aux_stop_ratio=args.moe_aux_stop_ratio,
    )
    
    # # Save tokenizer
    tokenizer.save_pretrained(os.path.join(run_dir, "tokenizer"))
    
    # Training
    if hasattr(model, "config"):
        model.config.use_cache = False
    trainer.train()
    
    # After training
    final_model_path = os.path.join(run_dir, "final_model")
    trainer.save_state()
    model.save_pretrained(final_model_path)

    
    return run_dir

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="LoRA training with organized output")
    
    # Dataset arguments
    parser.add_argument("--data_path", type=str, default="data/commonsense/commonsense_170k.json", help="Path to the training data")
    parser.add_argument('--train_on_inputs', action='store_true', help='Train on inputs')

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B", help="Model name")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="HF cache directory for model/tokenizer weights")
    parser.add_argument("--hf_fast_download", action="store_true", help="Enable accelerated HF Hub download and parallel loading")
    parser.add_argument("--hf_preload", action="store_true", help="Pre-download full model snapshot before from_pretrained")
    parser.add_argument("--hf_download_workers", type=int, default=16, help="Number of workers for snapshot pre-download")
    parser.add_argument("--hf_parallel_loading_workers", type=int, default=8, help="Number of workers for parallel HF loading")
    parser.add_argument("--hf_prefer_safetensors", action="store_true", help="Prefer safetensors and skip .bin/.pth files during preload when possible")
    parser.add_argument("--hf_local_files_only", action="store_true", help="Load model/tokenizer only from local cache (no network)")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA R value")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha value")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout value")
    parser.add_argument("--peft_method", type=str, default="abba", choices=["abba", "moe_lora"], help="PEFT method to train")
    parser.add_argument("--moe_r_max", type=int, default=32, help="Rank-MoE maximum rank components (r_max)")
    parser.add_argument("--moe_top_k", type=int, default=1, help="Top-k routed experts per token for MoE-LoRA")
    parser.add_argument("--moe_router_hidden_dim", type=int, default=128, help="Hidden dim for router MLP (set 0 for linear router)")
    parser.add_argument("--moe_router_norm_type", type=str, default="layernorm", choices=["layernorm", "rmsnorm", "none"], help="Normalization in MoE router")
    parser.add_argument("--moe_router_activation", type=str, default="gelu", choices=["gelu", "silu", "relu"], help="Activation in MoE router MLP")
    parser.add_argument("--moe_entropy_loss_weight", type=float, default=0.001, help="Weight for MoE routing entropy regularizer")
    parser.add_argument("--moe_load_balance_loss_weight", type=float, default=0.001, help="Weight for MoE rank load balancing regularizer")
    parser.add_argument("--moe_mask_init_strategy", type=str, default="sigmoid", choices=["sigmoid", "xavier_norm"], help="Mask init strategy: sigmoid logits or xavier-initialized normalized masks")
    parser.add_argument("--moe_mask_init_value", type=float, default=0.9, help="Initial sigmoid value for MoE mask logits")
    parser.add_argument("--moe_mask_init_std", type=float, default=0.0, help="Std-dev noise added to initial MoE mask logits")
    parser.add_argument("--moe_aux_loss_cap", type=float, default=0.2, help="Cap total MoE aux contribution as a fraction of base loss (0 disables)")
    parser.add_argument("--moe_aux_warmup_ratio", type=float, default=0.1, help="Warm up MoE aux loss weights linearly over this fraction of training")
    parser.add_argument("--moe_aux_stop_ratio", type=float, default=0.6, help="Disable MoE aux losses after this training progress fraction")
    parser.add_argument("--moe_router_lr", type=float, default=1e-4, help="Router learning rate (often lower than LoRA LR for stable routing)")
    parser.add_argument("--moe_lora_weight_decay", type=float, default=0.01, help="Weight decay for LoRA/MoE adapter matrices")
    parser.add_argument("--moe_router_weight_decay", type=float, default=0.0, help="Weight decay for router parameters")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")
    parser.add_argument("--grad_acc_steps", type=int, default=24, help="Gradient accumulation steps")

    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--scheduler", type=str, default="linear", help="Learning rate scheduler")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable model-wide gradient checkpointing")
    parser.add_argument("--warmup_ratio", type=float, default=0.02, help="Warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Gradient clipping norm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, help="DataLoader worker count for faster input pipeline")
    
    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Run training
    run_dir = finetune()
