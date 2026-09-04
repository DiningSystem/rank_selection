import json
import os
import types
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback


@dataclass
class SoRAConfig:
    rmax: int = 8
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    target_modules: Optional[Union[List[str], str]] = None
    gate_lr: float = 0.1
    lambda_sparsity: float = 0.1

    def __post_init__(self):
        if self.target_modules is None:
            raise ValueError("target_modules cannot be None")


class SoRALayer(nn.Module):
    def __init__(self, base_layer: nn.Module, rmax: int = 8, lora_alpha: float = 8.0, lora_dropout: float = 0.0):
        super().__init__()
        if not hasattr(base_layer, "weight"):
            raise ValueError("Layer doesn't have a weight attribute")
        self.base_layer = base_layer
        self.out_features, self.in_features = base_layer.weight.shape
        self.rmax = rmax
        self.scaling = lora_alpha / max(rmax, 1)
        self.dropout = nn.Dropout(lora_dropout)
        self.merged = False
        self.disable_adapters = False

        weight = base_layer.weight
        self.lora_A = nn.Parameter(torch.empty(rmax, self.in_features, device=weight.device, dtype=weight.dtype))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rmax, device=weight.device, dtype=weight.dtype))
        self.gate = nn.Parameter(torch.ones(rmax, device=weight.device, dtype=weight.dtype))
        self.reset_parameters()
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base = self.base_layer(x)
        if self.disable_adapters or self.merged:
            return base
        adapter_input = self.dropout(x).to(self.lora_A.dtype)
        h = F.linear(adapter_input, self.lora_A)
        h = h * self.gate.to(h.dtype)
        delta = F.linear(h, self.lora_B)
        return base + (self.scaling * delta).to(base.dtype)

    def effective_rank(self):
        return int((self.gate.detach() != 0).sum().item())

    def merge(self):
        if self.merged:
            return
        keep = self.gate.detach() != 0
        if keep.any():
            gated_B = self.lora_B[:, keep] * self.gate[keep].unsqueeze(0)
            delta = (gated_B @ self.lora_A[keep, :]) * self.scaling
            self.base_layer.weight.data.add_(delta.to(self.base_layer.weight.dtype))
        self.merged = True


@torch.no_grad()
def proximal_gate_update(gate, gate_grad, gate_lr, lambda_sparsity):
    gate.add_(gate_grad, alpha=-gate_lr)
    threshold = gate_lr * lambda_sparsity
    gate.copy_(torch.sign(gate) * torch.clamp(gate.abs() - threshold, min=0.0))


def get_submodules(model, key):
    if "." in key:
        return key.rsplit(".", 1)
    return "", key


def apply_sora(model, config: SoRAConfig):
    target_modules = config.target_modules if isinstance(config.target_modules, list) else [config.target_modules]
    model.sora_config = config
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(target == name.split(".")[-1] or target in name for target in target_modules):
            parent_name, target_name = get_submodules(model, name)
            parent = model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, target_name, SoRALayer(module, config.rmax, config.lora_alpha, config.lora_dropout))
    model.mark_only_sora_as_trainable = types.MethodType(mark_only_sora_as_trainable, model)
    model.save_pretrained = types.MethodType(save_pretrained, model)
    model.merge_and_unload = types.MethodType(merge_and_unload, model)
    model.mark_only_sora_as_trainable()
    return model


def mark_only_sora_as_trainable(self):
    for _, param in self.named_parameters():
        param.requires_grad = False
    for module in self.modules():
        if isinstance(module, SoRALayer):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            module.gate.requires_grad = True


def save_pretrained(self, save_directory, **kwargs):
    os.makedirs(save_directory, exist_ok=True)
    if hasattr(self, "sora_config"):
        with open(os.path.join(save_directory, "adapter_config.json"), "w") as f:
            json.dump(self.sora_config.__dict__, f, indent=2)
    state = {}
    for name, module in self.named_modules():
        if isinstance(module, SoRALayer):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
            state[f"{name}.gate"] = module.gate.detach().cpu()
    torch.save(state, os.path.join(save_directory, "adapter_model.bin"))


def set_sora_state_dict(model, adapter_state_dict: Dict[str, torch.Tensor]):
    for name, module in model.named_modules():
        if isinstance(module, SoRALayer):
            for attr in ("lora_A", "lora_B", "gate"):
                key = f"{name}.{attr}"
                if key in adapter_state_dict:
                    getattr(module, attr).data.copy_(adapter_state_dict[key].to(getattr(module, attr).device, getattr(module, attr).dtype))


def merge_and_unload(self):
    for module in self.modules():
        if isinstance(module, SoRALayer):
            module.merge()
    return self


class SoRAGateCallback(TrainerCallback):
    def __init__(self, config: SoRAConfig):
        self.config = config

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        ranks = []
        for module in model.modules():
            if isinstance(module, SoRALayer) and module.gate.grad is not None:
                proximal_gate_update(module.gate, module.gate.grad, self.config.gate_lr, self.config.lambda_sparsity)
                module.gate.grad = None
                ranks.append(module.effective_rank())
        if ranks:
            logs = {"sora/effective_rank_mean": sum(ranks) / len(ranks), "sora/effective_rank_min": min(ranks), "sora/effective_rank_max": max(ranks)}
            try:
                kwargs.get("trainer").log(logs)
            except Exception:
                pass
        return control


class SoRATrainer(Trainer):
    def __init__(self, *args, sora_config: Optional[SoRAConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sora_config = sora_config or getattr(self.model, "sora_config", None)
        if self.sora_config is not None:
            self.add_callback(SoRAGateCallback(self.sora_config))
