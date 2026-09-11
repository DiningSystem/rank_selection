import json
import os
import types
from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import Trainer, TrainerCallback


@dataclass
class EvolveLoRAConfig:
    r_max: int = 32
    shared_rank: int = 8
    r_min: int = 2
    evolve_rank_delay_ratio: float = 0.15
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
    router_hidden_dim: int = 64
    ortho_weight: float = 1e-4
    active_component_threshold: float = 0.1
    active_log_max_layers: int = 0
    active_log_seed: int = 42

    def __post_init__(self):
        if self.target_modules is None:
            raise ValueError("target_modules cannot be None")
        if not 0 <= self.shared_rank < self.r_max:
            raise ValueError("shared_rank must be non-negative and smaller than r_max")

    @property
    def routed_rank(self) -> int:
        """Number of input-routed spectral directions."""
        return self.r_max - self.shared_rank


class SpectralLoRALayer(nn.Module):
    """Shared plus input-routed spectral LoRA.

    The shared adapter is ``U_s diag(lambda_s) V_s^T`` and is active for
    every input.  The routed adapter is ``U_r diag(lambda_r(x)) V_r^T``.
    Neither factor is constrained to be orthogonal.
    """

    def __init__(self, base_layer: nn.Module, r_max: int = 32, alpha: float = 16.0,
                 dropout: float = 0.0, gate_floor: float = 0.05,
                 detach_router_input: bool = True, router_hidden_dim: int = 64,
                 shared_rank: int = 8):
        super().__init__()
        if not hasattr(base_layer, "weight"):
            raise ValueError("Layer doesn't have a weight attribute")
        self.base_layer = base_layer
        self.out_features, self.in_features = base_layer.weight.shape
        self.r_max = r_max
        if not 0 <= shared_rank < r_max:
            raise ValueError("shared_rank must be non-negative and smaller than r_max")
        self.shared_rank = shared_rank
        self.routed_rank = r_max - shared_rank
        self.alpha = alpha
        self.scaling = alpha / max(r_max, 1)
        self.gate_floor = gate_floor
        self.detach_router_input = detach_router_input
        hidden_dim = min (self.in_features // 2, router_hidden_dim)
        hidden_dim = max(1, hidden_dim)
        weight = base_layer.weight
        self.adapter_dtype = weight.dtype
        adapter_device = weight.device
        self.log_temperature = nn.Parameter(torch.zeros(()))

        W0 = self.base_layer.weight.detach().float()

        U_svd, S_svd, Vh_svd = torch.linalg.svd(
            W0,
            full_matrices=False,
        )

        # Pad when a requested adapter rank exceeds the matrix rank.  This
        # keeps adapters usable for small Linear layers as well as LLM layers.
        U_init = torch.zeros(self.out_features, self.r_max, device=W0.device, dtype=W0.dtype)
        V_init = torch.zeros(self.in_features, self.r_max, device=W0.device, dtype=W0.dtype)
        available_rank = min(self.r_max, S_svd.numel())
        U_init[:, :available_rank] = U_svd[:, :available_rank]
        V_init[:, :available_rank] = Vh_svd[:available_rank, :].T
        

        # U_svd = U_svd[:, :r_max]
        # S_svd = S_svd[:r_max]
        # V_svd = Vh_svd[:r_max, :].T

        # S_sqrt = torch.sqrt(S_svd)

        # U_init = U_svd * S_sqrt.unsqueeze(0)
        # V_init = V_svd * S_sqrt.unsqueeze(0)


        #self.U = nn.Parameter(torch.randn(self.out_features, r_max, device=adapter_device, dtype=self.adapter_dtype) * 0.02) 
        #self.V = nn.Parameter(torch.randn(self.in_features, r_max, device=adapter_device, dtype=self.adapter_dtype) * 0.02)
        self.U_shared = nn.Parameter(
            U_init[:, :self.shared_rank].to(
                device=adapter_device,
                dtype=self.adapter_dtype,
            )
        )
        self.V_shared = nn.Parameter(
            V_init[:, :self.shared_rank].to(
                device=adapter_device,
                dtype=self.adapter_dtype,
            )
        )
        self.U_routed = nn.Parameter(U_init[:, self.shared_rank:].to(device=adapter_device, dtype=self.adapter_dtype))
        self.V_routed = nn.Parameter(V_init[:, self.shared_rank:].to(device=adapter_device, dtype=self.adapter_dtype))
        # These coefficients are deliberately independent of x, unlike the
        # routed lambda values below.
        self.shared_lambdas = nn.Parameter(torch.ones(self.shared_rank, device=adapter_device, dtype=self.adapter_dtype))

        self.router = nn.Sequential(
            #nn.Linear(self.in_features, hidden_dim, device=adapter_device, dtype=self.adapter_dtype),
            #nn.GELU(),
            nn.Linear(self.in_features, self.routed_rank, device=adapter_device, dtype=self.adapter_dtype),
        )
        self.dropout = nn.Dropout(dropout)
        self.merged = False
        self.disable_adapters = False
        self.last_lambdas = None
        self.last_router_probs = None

    def forward(self, x):
        y = self.base_layer(x)
        if self.disable_adapters or self.merged:
            self.last_lambdas = None
            return y
        adapter_input = x.to(self.adapter_dtype)
        router_input = adapter_input.detach() if self.detach_router_input else adapter_input
        dropped = self.dropout(adapter_input)
        shared_spectral = (dropped @ self.V_shared) * self.shared_lambdas
        shared_out = shared_spectral @ self.U_shared.t()
        routed_lambdas = torch.softmax(self.router(router_input), dim=-1)
        self.last_router_probs = routed_lambdas.float()
        # All routing/rank metrics intentionally apply only to routed experts.
        self.last_lambdas = routed_lambdas.float()
        routed_spectral = (dropped @ self.V_routed) * routed_lambdas
        routed_out = routed_spectral @ self.U_routed.t()
        adapter_out = (shared_out + routed_out) * self.scaling
        return y + adapter_out.to(y.dtype)

    def merge(self):
        raise RuntimeError("Evolve-LoRA is input-conditioned and cannot be exactly merged into static weights.")


def orthogonal_loss(U, V):
    U = U.float()
    V = V.float()

    Iu = torch.eye(U.shape[1], device=U.device)
    Iv = torch.eye(V.shape[1], device=V.device)

    return (
        ((U.T @ U - Iu) ** 2).mean()
        +
        ((V.T @ V - Iv) ** 2).mean()
    )
def expert_balance_loss(
    router_probs,
    eps=1e-8,
):
    """
    Encourage diverse expert utilization across the batch.

    Given:

        p_{b,i}

    define:

        p_bar_i = mean_b p_{b,i}

    and minimize:

        KL(p_bar || Uniform).

    This prevents the same expert from winning for
    essentially every sample.
    """

    p = router_probs.reshape(
        -1,
        router_probs.shape[-1],
    )

    mean_p = p.mean(dim=0)

    r = mean_p.shape[0]

    uniform = torch.full_like(
        mean_p,
        1.0 / r,
    )

    kl = (
        mean_p
        * torch.log(
            (mean_p + eps)
            / uniform
        )
    ).sum()

    return kl

def get_submodules(model, key):
    if "." in key:
        return key.rsplit(".", 1)
    return "", key


def mark_only_evolve_lora_as_trainable(self):
    for _, param in self.named_parameters():
        param.requires_grad = False
    for module in self.modules():
        if isinstance(module, SpectralLoRALayer):
            module.U_shared.requires_grad = True
            module.V_shared.requires_grad = True
            module.U_routed.requires_grad = True
            module.V_routed.requires_grad = True
            module.shared_lambdas.requires_grad = True
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
            state[f"{name}.U_shared"] = module.U_shared.detach().cpu()
            state[f"{name}.V_shared"] = module.V_shared.detach().cpu()
            state[f"{name}.shared_lambdas"] = module.shared_lambdas.detach().cpu()
            state[f"{name}.U_routed"] = module.U_routed.detach().cpu()
            state[f"{name}.V_routed"] = module.V_routed.detach().cpu()
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
                                                          config.gate_floor, config.detach_router_input,
                                                          config.router_hidden_dim, config.shared_rank))
            model.evolve_lora_layers.add(name)
    model.mark_only_evolve_lora_as_trainable = types.MethodType(mark_only_evolve_lora_as_trainable, model)
    model.save_pretrained = types.MethodType(save_pretrained, model)
    model.mark_only_evolve_lora_as_trainable()
    return model


def set_evolve_lora_state_dict(model, adapter_state_dict: Dict[str, torch.Tensor]):
    for name, module in model.named_modules():
        if isinstance(module, SpectralLoRALayer):
            for parameter_name in ("U_shared", "V_shared", "shared_lambdas", "U_routed", "V_routed"):
                key = f"{name}.{parameter_name}"
                parameter = getattr(module, parameter_name)
                if key in adapter_state_dict:
                    parameter.data.copy_(adapter_state_dict[key].to(parameter.device, parameter.dtype))
            prefix = f"{name}.router."
            for i, layer in enumerate(module.router):
                params = list(layer.parameters())
                if not params:
                    continue
                sd = {k.rsplit('.', 1)[-1]: v.to(params[0].device, params[0].dtype)
                      for k, v in adapter_state_dict.items() if k.startswith(f"{prefix}{i}.")}
                if sd:
                    layer.load_state_dict(sd, strict=False)


def effective_rank(lambdas):
    probs = lambdas #/ (lambdas.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    return torch.exp(entropy)

def router_js_diversity_loss(lambdas, eps=1e-8):
    """
    Encourage different inputs to have different routing distributions
    using Jensen-Shannon divergence.

    lambdas:
        [N, routed_rank] or [B, T, routed_rank]
    """
    if lambdas.dim() == 3:
        lambdas = lambdas.reshape(-1, lambdas.size(-1))

    # Normalize routing weights
    p = lambdas / (lambdas.sum(dim=-1, keepdim=True) + eps)
    p = p.clamp_min(eps)

    # Average routing distribution
    q = p.mean(dim=0, keepdim=True)
    q = q.clamp_min(eps)

    # Mixture distribution
    m = 0.5 * (p + q)

    # JS(p || q)
    kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=-1)
    kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=-1)

    js = 0.5 * (kl_pm + kl_qm)

    # Maximize JS => minimize negative JS
    return -js.mean()

def routing_entropy(router_probs, eps=1e-8):
    p = router_probs / (router_probs.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(p * torch.log(p + eps)).sum(dim=-1)
    return entropy


def entropy_floor_loss(router_probs, target_entropy):
    entropy = routing_entropy(router_probs)

    return torch.relu(
        target_entropy - entropy
    ).pow(2).mean()

def rank_regularizer_weight(step, start_step, alpha_max=0.01):
    return alpha_max if step >= start_step else 0.0


class EvolveLoRALogCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        self.trainer._flush_evolve_logs()
        return control


class EvolveLoRATrainer(Trainer):
    def __init__(self, *args, evolve_lora_config: Optional[EvolveLoRAConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.evolve_lora_config = evolve_lora_config or getattr(self.model, "evolve_lora_config", None)
        self._active_log_layer_names = None
        self._pending_evolve_logs = None
        self._pending_evolve_log_count = 0
        self.add_callback(EvolveLoRALogCallback(self))

    def _sanitize_log_key(self, name):
        return name.replace(".", "/")

    def _get_active_log_layer_names(self, model):
        if self._active_log_layer_names is not None:
            return self._active_log_layer_names

        layer_names = [
            name for name, module in model.named_modules()
            if isinstance(module, SpectralLoRALayer)
        ]
        cfg = self.evolve_lora_config
        max_layers = getattr(cfg, "active_log_max_layers", 0) if cfg is not None else 0
        if max_layers and max_layers > 0 and len(layer_names) > max_layers:
            rng = random.Random(getattr(cfg, "active_log_seed", 42))
            layer_names = sorted(rng.sample(layer_names, max_layers))

        self._active_log_layer_names = set(layer_names)
        return self._active_log_layer_names

    def _accumulate_evolve_logs(self, logs):
        if not logs:
            return
        if self._pending_evolve_logs is None:
            self._pending_evolve_logs = {key: float(value) for key, value in logs.items()}
        else:
            for key, value in logs.items():
                self._pending_evolve_logs[key] = self._pending_evolve_logs.get(key, 0.0) + float(value)
        self._pending_evolve_log_count += 1

    def _flush_evolve_logs(self):
        if not self._pending_evolve_logs or self._pending_evolve_log_count == 0:
            return
        count = self._pending_evolve_log_count
        logs = {key: value / count for key, value in self._pending_evolve_logs.items()}
        self._pending_evolve_logs = None
        self._pending_evolve_log_count = 0
        self.log(logs)

    def _collect_lambdas(self):
        vals = [m.last_lambdas.reshape(-1, m.routed_rank) for m in self.model.modules()
                if isinstance(m, SpectralLoRALayer) and m.last_lambdas is not None]
        return torch.cat(vals, dim=0) if vals else None

    def _active_component_logs(self, model):
        cfg = self.evolve_lora_config
        if cfg is None:
            return {}

        threshold = getattr(cfg, "active_component_threshold", 0.1)
        selected_layers = self._get_active_log_layer_names(model)
        logs = {}
        layer_active_counts = []
        for name, module in model.named_modules():
            if not isinstance(module, SpectralLoRALayer) or module.last_lambdas is None:
                continue

            active_counts = (module.last_lambdas.float() > threshold).sum(dim=-1).float()
            mean_active = active_counts.mean().detach()
            layer_active_counts.append(mean_active)
            if name in selected_layers:
                logs[f"evolve/active_components/{self._sanitize_log_key(name)}"] = mean_active.item()

        if layer_active_counts:
            logs["evolve/active_components_mean"] = torch.stack(layer_active_counts).mean().item()
            logs["evolve/active_component_threshold"] = float(threshold)
        return logs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        orth_losses = []
        entropy_losses = []
        balance_losses = []

        # for module in model.modules():
        #     if isinstance(module, SpectralLoRALayer):
        #         orth_losses.append(orthogonal_loss(module.U, module.V))

        # orth_loss = torch.stack(
        #     orth_losses
        # ).mean()
        task_loss = outputs.loss
        cfg = self.evolve_lora_config
        lambdas = self._collect_lambdas()
        if cfg is None or lambdas is None or not hasattr(outputs, "logits"):
            return (task_loss, outputs) if return_outputs else task_loss
        erank = effective_rank(lambdas.float())
        rank_delay_step = int(self.state.max_steps * cfg.evolve_rank_delay_ratio)
        alpha_t = rank_regularizer_weight(self.state.global_step, rank_delay_step, cfg.alpha_max)
        rank_reg = erank.mean()
        diversity = router_js_diversity_loss(lambdas=lambdas.float())
        #ent_loss = entropy_floor_loss(lambdas.float(), 0.35).mean()
        loss = task_loss.float() #+ alpha_t * diversity#((rank_reg-1)/(cfg.r_max-1))   #+ \
            #cfg.ortho_weight * orth_loss #+ cfg.beta * balance_loss
        if model.training:
            logs = {
                # Retain the old key for existing dashboards; it now has the
                # explicitly routed-only meaning documented by the new key.
                "evolve/erank": rank_reg.detach().item(),
                "evolve/routed_effective_rank": rank_reg.detach().item(),
                "evolve/shared_rank": float(cfg.shared_rank),
                "evolve/routed_rank": float(cfg.routed_rank),
                "loss": task_loss.float(),
                "total_loss": loss,
            }
            logs.update(self._active_component_logs(model))
            self._accumulate_evolve_logs(logs)
        return (loss, outputs) if return_outputs else loss
