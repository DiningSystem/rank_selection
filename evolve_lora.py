import json
import math
import os
import types
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


@dataclass
class EvolveLoRAConfig:
    r_max: int = 32
    r_min: int = 2
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Optional[Union[List[str], str]] = None
    gate_floor: float = 0.05
    detach_router_input: bool = True
    beta: float = 0.05
    alpha_max: float = 0.01
    anneal_k: float = 5e-5
    complexity_ema: float = 0.9
    bias: str = "none"
    modules_to_save: Optional[List[str]] = None

    def __post_init__(self):
        if self.target_modules is None:
            raise ValueError("target_modules cannot be None")


class SpectralLoRALayer(nn.Module):
    """Input-conditioned spectral LoRA: ΔW(x)=U diag(lambda(x)) V^T."""

    def __init__(self, base_layer: nn.Module, r_max: int = 32, alpha: float = 16.0,
                 dropout: float = 0.0, gate_floor: float = 0.05,
                 detach_router_input: bool = True):
        super().__init__()
        if not hasattr(base_layer, "weight"):
            raise ValueError("Layer doesn't have a weight attribute")
        self.base_layer = base_layer
        self.out_features, self.in_features = base_layer.weight.shape
        self.r_max = r_max
        self.alpha = alpha
        self.scaling = alpha / max(r_max, 1)
        self.gate_floor = gate_floor
        self.detach_router_input = detach_router_input
        hidden_dim = max(1, self.in_features // 2)
        self.U = nn.Parameter(torch.randn(self.out_features, r_max) * 0.02)
        self.V = nn.Parameter(torch.randn(self.in_features, r_max) * 0.02)
        self.router = nn.Sequential(
            nn.Linear(self.in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, r_max),
        )
        self.dropout = nn.Dropout(dropout)
        self.merged = False
        self.disable_adapters = False
        self.last_lambdas = None

    def forward(self, x):
        y = self.base_layer(x)
        if self.disable_adapters or self.merged:
            self.last_lambdas = None
            return y
        router_input = x.detach() if self.detach_router_input else x
        lambdas = self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(self.router(router_input))
        self.last_lambdas = lambdas
        dropped = self.dropout(x)
        spectral = (dropped @ self.V) * lambdas
        return y + (spectral @ self.U.t()) * self.scaling

    def merge(self):
        raise RuntimeError("Evolve-LoRA is input-conditioned and cannot be exactly merged into static weights.")


def get_submodules(model, key):
    if "." in key:
        return key.rsplit(".", 1)
    return "", key


def mark_only_evolve_lora_as_trainable(self):
    for _, param in self.named_parameters():
        param.requires_grad = False
    for module in self.modules():
        if isinstance(module, SpectralLoRALayer):
            module.U.requires_grad = True
            module.V.requires_grad = True
            for param in module.router.parameters():
                param.requires_grad = True


def save_pretrained(self, save_directory, **kwargs):
    os.makedirs(save_directory, exist_ok=True)
    if hasattr(self, "evolve_lora_config"):
        with open(os.path.join(save_directory, "adapter_config.json"), "w") as f:
            json.dump(self.evolve_lora_config.__dict__, f, indent=2)
    state = {}
    for name, module in self.named_modules():
        if isinstance(module, SpectralLoRALayer):
            state[f"{name}.U"] = module.U.detach().cpu()
            state[f"{name}.V"] = module.V.detach().cpu()
            for i, layer in enumerate(module.router):
                if hasattr(layer, "state_dict"):
                    for k, v in layer.state_dict().items():
                        state[f"{name}.router.{i}.{k}"] = v.detach().cpu()
    torch.save(state, os.path.join(save_directory, "adapter_model.bin"))


def apply_evolve_lora(model, config: EvolveLoRAConfig):
    target_modules = config.target_modules if isinstance(config.target_modules, list) else [config.target_modules]
    model.evolve_lora_config = config
    model.evolve_lora_layers = set()
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(target == name.split(".")[-1] or target in name for target in target_modules):
            parent_name, target_name = get_submodules(model, name)
            parent = model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, target_name, SpectralLoRALayer(module, config.r_max, config.alpha, config.dropout,
                                                          config.gate_floor, config.detach_router_input))
            model.evolve_lora_layers.add(name)
    model.mark_only_evolve_lora_as_trainable = types.MethodType(mark_only_evolve_lora_as_trainable, model)
    model.save_pretrained = types.MethodType(save_pretrained, model)
    model.mark_only_evolve_lora_as_trainable()
    return model


def set_evolve_lora_state_dict(model, adapter_state_dict: Dict[str, torch.Tensor]):
    for name, module in model.named_modules():
        if isinstance(module, SpectralLoRALayer):
            if f"{name}.U" in adapter_state_dict:
                module.U.data.copy_(adapter_state_dict[f"{name}.U"].to(module.U.device, module.U.dtype))
            if f"{name}.V" in adapter_state_dict:
                module.V.data.copy_(adapter_state_dict[f"{name}.V"].to(module.V.device, module.V.dtype))
            prefix = f"{name}.router."
            for i, layer in enumerate(module.router):
                params = list(layer.parameters())
                if not params:
                    continue
                sd = {k.rsplit('.', 1)[-1]: v.to(params[0].device, params[0].dtype)
                      for k, v in adapter_state_dict.items() if k.startswith(f"{prefix}{i}.")}
                if sd:
                    layer.load_state_dict(sd, strict=False)


def compute_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-8)).sum(dim=-1)


def sequence_complexity(logits):
    return compute_entropy(logits).mean(dim=-1)


def effective_rank(lambdas):
    probs = lambdas / (lambdas.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    return torch.exp(entropy)


def target_rank(complexity, r_min, r_max):
    c_min, c_max = complexity.min().detach(), complexity.max().detach()
    norm_c = (complexity - c_min) / (c_max - c_min + 1e-8)
    return r_min + (r_max - r_min) * norm_c


def anneal_alpha(step, alpha_max=0.01, k=5e-5):
    return alpha_max * (1 - math.exp(-k * step))


class EvolveLoRATrainer(Trainer):
    def __init__(self, *args, evolve_lora_config: Optional[EvolveLoRAConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.evolve_lora_config = evolve_lora_config or getattr(self.model, "evolve_lora_config", None)
        self._complexity_ema = None

    def _collect_lambdas(self):
        vals = [m.last_lambdas.reshape(-1, m.r_max) for m in self.model.modules()
                if isinstance(m, SpectralLoRALayer) and m.last_lambdas is not None]
        return torch.cat(vals, dim=0) if vals else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        task_loss = outputs.loss
        cfg = self.evolve_lora_config
        lambdas = self._collect_lambdas()
        if cfg is None or lambdas is None or not hasattr(outputs, "logits"):
            return (task_loss, outputs) if return_outputs else task_loss
        with torch.no_grad():
            complexity = sequence_complexity(outputs.logits.detach()).reshape(-1)
            current = complexity.mean()
            self._complexity_ema = current if self._complexity_ema is None else cfg.complexity_ema * self._complexity_ema + (1 - cfg.complexity_ema) * current
            smoothed = complexity + (self._complexity_ema - current)
        erank = effective_rank(lambdas)
        if erank.numel() != smoothed.numel():
            smoothed = smoothed.mean().expand_as(erank)
        info_loss = F.mse_loss(erank, target_rank(smoothed, cfg.r_min, cfg.r_max))
        alpha_t = anneal_alpha(self.state.global_step, cfg.alpha_max, cfg.anneal_k)
        rank_reg = erank.mean()
        loss = task_loss + alpha_t * rank_reg + cfg.beta * info_loss
        self.log({"evolve/erank": rank_reg.detach().item(), "evolve/info_loss": info_loss.detach().item(), "evolve/alpha": alpha_t})
        return (loss, outputs) if return_outputs else loss
