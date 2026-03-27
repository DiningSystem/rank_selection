import types
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file


class RankRouter(nn.Module):
    """Hypernetwork router over rank-1 components.

    g(x) in R^{batch x r_max}
    """

    def __init__(self, d_model: int, r_max: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, r_max),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RankMoELoRALayer(nn.Module):
    """Rank-MoE LoRA layer with masked rank-1 components.

    Rank component i behaves like an expert:
      (b_i ⊙ m_i^(b)) (a_i ⊙ m_i^(a))^T

    ΔW(x)x is computed without materializing full ΔW(x).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        r_max: int,
        top_k: int = 1,
        router_hidden_dim: int = 128,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if r_max <= 0:
            raise ValueError(f"r_max must be > 0, got {r_max}")
        if top_k <= 0:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        self.d_in = d_in
        self.d_out = d_out
        self.r_max = r_max
        self.top_k = min(top_k, r_max)

        self.base = nn.Linear(d_in, d_out, bias=bias)
        if freeze_base:
            self.base.weight.requires_grad = False
            if self.base.bias is not None:
                self.base.bias.requires_grad = False

        # Shared low-rank parameters.
        self.B = nn.Parameter(torch.empty(d_out, r_max))
        self.A = nn.Parameter(torch.empty(r_max, d_in))

        # Mask logits; masks are applied per-rank-component before multiplication.
        self.S_b = nn.Parameter(torch.zeros(d_out, r_max))
        self.S_a = nn.Parameter(torch.zeros(r_max, d_in))

        self.router = RankRouter(d_model=d_in, r_max=r_max, hidden_dim=router_hidden_dim)

        self.register_buffer("rank_usage_counts", torch.zeros(r_max), persistent=False)
        self.register_buffer("routed_token_count", torch.tensor(0.0), persistent=False)
        self._last_g: Optional[torch.Tensor] = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.B, mean=0.0, std=0.02)
        nn.init.normal_(self.A, mean=0.0, std=0.02)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r_max: int,
        top_k: int = 1,
        router_hidden_dim: int = 128,
        freeze_base: bool = True,
    ) -> "RankMoELoRALayer":
        layer = cls(
            d_in=linear.in_features,
            d_out=linear.out_features,
            r_max=r_max,
            top_k=top_k,
            router_hidden_dim=router_hidden_dim,
            bias=linear.bias is not None,
            freeze_base=freeze_base,
        )
        layer = layer.to(device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            layer.base.weight.copy_(linear.weight)
            if linear.bias is not None and layer.base.bias is not None:
                layer.base.bias.copy_(linear.bias)
        return layer

    def _flatten_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        original_shape = x.shape
        return x.reshape(-1, x.shape[-1]), original_shape

    def _topk_normalize(self, g: torch.Tensor) -> torch.Tensor:
        if self.top_k >= self.r_max:
            return g
        top_vals, top_idx = torch.topk(g, k=self.top_k, dim=-1)
        masked = torch.zeros_like(g)
        masked.scatter_(1, top_idx, top_vals)
        denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return masked / denom

    def masked_factors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        m_b = torch.sigmoid(self.S_b)
        m_a = torch.sigmoid(self.S_a)
        b_tilde = self.B * m_b
        a_tilde = self.A * m_a
        return a_tilde, b_tilde, m_a, m_b

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x_2d, original_shape = self._flatten_tokens(x)
        base_in = x_2d if x_2d.dtype == self.base.weight.dtype else x_2d.to(self.base.weight.dtype)
        base_y = self.base(base_in)

        a_tilde, b_tilde, m_a, m_b = self.masked_factors()

        # Hypernetwork routing over rank components.
        router_dtype = next(self.router.parameters()).dtype
        router_in = x_2d if x_2d.dtype == router_dtype else x_2d.to(router_dtype)
        g_logits = self.router(router_in)
        g = F.softmax(g_logits, dim=-1)
        g = self._topk_normalize(g)
        if return_aux:
            self._last_g = g
        elif self.training:
            self._last_g = g.detach()
        else:
            self._last_g = None

        # Efficient residual path (no ΔW materialization):
        # Ax: [batch, r_max], weighted: [batch, r_max], delta: [batch, d_out]
        ax_in = x_2d if x_2d.dtype == a_tilde.dtype else x_2d.to(a_tilde.dtype)
        ax = F.linear(ax_in, a_tilde)
        weighted = g.to(ax.dtype) * ax
        if weighted.dtype != b_tilde.dtype:
            weighted = weighted.to(b_tilde.dtype)
        delta = F.linear(weighted, b_tilde)
        if delta.dtype != base_y.dtype:
            delta = delta.to(base_y.dtype)
        if (not self.training) and (not torch.is_grad_enabled()):
            base_y.add_(delta)
            y = base_y.reshape(*original_shape[:-1], self.d_out)
        else:
            y = (base_y + delta).reshape(*original_shape[:-1], self.d_out)

        with torch.no_grad():
            self.rank_usage_counts += (g > 0).sum(dim=0).to(self.rank_usage_counts.dtype)
            self.routed_token_count += float(x_2d.size(0))

        if not return_aux:
            return y

        aux = {
            "rank_sparsity": self.rank_sparsity_loss(g),
            "mask_l1": self.mask_sparsity_loss(m_a, m_b),
            "avg_active_rank": self.average_active_rank(g),
            "g_distribution": g.mean(dim=0),
            "mask_mean": {"mask_a_mean": m_a.mean(), "mask_b_mean": m_b.mean()},
        }
        return y, aux

    def rank_sparsity_loss(self, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        g_ = self._last_g if g is None else g
        if g_ is None:
            raise RuntimeError("rank_sparsity_loss requires forward pass or explicit g")
        return g_.sum(dim=-1).mean()

    def mask_sparsity_loss(
        self,
        m_a: Optional[torch.Tensor] = None,
        m_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if m_a is None or m_b is None:
            _, _, m_a, m_b = self.masked_factors()
        return m_a.abs().sum() + m_b.abs().sum()

    def average_active_rank(self, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        g_ = self._last_g if g is None else g
        if g_ is None:
            raise RuntimeError("average_active_rank requires forward pass or explicit g")
        return (g_ > 0).float().sum(dim=-1).mean()

    def rank_usage_frequency(self) -> torch.Tensor:
        denom = torch.clamp(self.routed_token_count, min=1.0)
        return self.rank_usage_counts / denom

    def debug_rank1_equivalence(self) -> torch.Tensor:
        a_tilde, b_tilde, _, _ = self.masked_factors()
        delta_direct = b_tilde @ a_tilde
        delta_rank1 = torch.einsum("or,ri->oi", b_tilde, a_tilde)
        return (delta_direct - delta_rank1).abs().max()


@dataclass
class MoELoRAConfig:
    experts_config: Sequence[Dict[str, int]]
    r_max: Optional[int] = None
    top_k: int = 1
    router_hidden_dim: Optional[int] = None
    target_modules: Optional[Sequence[str]] = None
    freeze_base: bool = True

    def __post_init__(self) -> None:
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")
        if not self.experts_config:
            raise ValueError("experts_config cannot be empty")
        if self.r_max is not None and self.r_max <= 0:
            raise ValueError("r_max must be > 0 when provided")

    @property
    def resolved_r_max(self) -> int:
        return self.r_max if self.r_max is not None else max(int(cfg["rank"]) for cfg in self.experts_config)


def _get_submodules(model: nn.Module, key: str):
    if "." in key:
        parent_name, target_name = key.rsplit(".", 1)
    else:
        parent_name, target_name = "", key
    return parent_name, target_name


def mark_only_moe_lora_as_trainable(self):
    for _, param in self.named_parameters():
        param.requires_grad = False

    for _, module in self.named_modules():
        if isinstance(module, RankMoELoRALayer):
            for param in module.parameters():
                if param is module.base.weight or param is module.base.bias:
                    continue
                param.requires_grad = True


def _with_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {f"{prefix}{k}": v for k, v in state_dict.items()}


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_moe_state_dict_flexible(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
    """Load state dict while handling optional leading `model.` prefix mismatch."""
    try:
        return self.load_state_dict(state_dict, strict=strict)
    except RuntimeError as err:
        model_keys = set(self.state_dict().keys())
        has_model_prefix = any(k.startswith("model.") for k in state_dict.keys())

        candidates = []
        if has_model_prefix:
            candidates.append(_strip_prefix(state_dict, "model."))
        else:
            candidates.append(_with_prefix(state_dict, "model."))

        for candidate in candidates:
            if not candidate:
                continue
            overlap = len(model_keys.intersection(candidate.keys()))
            if overlap == 0:
                continue
            return self.load_state_dict(candidate, strict=strict)
        raise err


def save_moe_pretrained(self, save_directory: str, **kwargs):
    """Save MoE checkpoint in eval-compatible key namespace."""
    state_dict = self.state_dict()
    return self._moe_original_save_pretrained(save_directory, state_dict=state_dict, **kwargs)


def merge_and_unload_moe_lora(self):
    raise NotImplementedError(
        "Rank-MoE-LoRA uses input-dependent routing and cannot be exactly merged into a static linear layer."
    )


def apply_moe_lora(model: nn.Module, config: MoELoRAConfig):
    model.peft_config = getattr(model, "peft_config", {})
    model.adapter_layers = getattr(model, "adapter_layers", set())
    model.peft_config["moe_lora"] = config

    replaced_count = 0
    target_modules = list(config.target_modules)
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            parent_name, target_name = _get_submodules(model, name)
            parent = model
            for name_part in parent_name.split("."):
                if name_part:
                    parent = getattr(parent, name_part)

            router_hidden = config.router_hidden_dim if config.router_hidden_dim is not None else 128
            wrapped = RankMoELoRALayer.from_linear(
                module,
                r_max=config.resolved_r_max,
                top_k=config.top_k,
                router_hidden_dim=router_hidden,
                freeze_base=config.freeze_base,
            )
            setattr(parent, target_name, wrapped)
            model.adapter_layers.add(name)
            replaced_count += 1

    if replaced_count == 0:
        raise RuntimeError("No target nn.Linear modules were replaced by Rank-MoE-LoRA.")

    model.mark_only_adapters_as_trainable = types.MethodType(mark_only_moe_lora_as_trainable, model)
    model.merge_and_unload = types.MethodType(merge_and_unload_moe_lora, model)
    model.load_moe_state_dict_flexible = types.MethodType(load_moe_state_dict_flexible, model)
    model._moe_original_save_pretrained = model.save_pretrained
    model.save_pretrained = types.MethodType(save_moe_pretrained, model)
    model.moe_lora_replaced_modules = replaced_count
    mark_only_moe_lora_as_trainable(model)
    return model


def get_moe_lora_model(model: nn.Module, config: MoELoRAConfig):
    return apply_moe_lora(model, config)


def load_moe_checkpoint_state_dict(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Load sharded HF safetensors checkpoint into a single state dict."""
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        single_path = os.path.join(checkpoint_dir, "model.safetensors")
        if not os.path.exists(single_path):
            raise FileNotFoundError(f"No safetensors checkpoint found under: {checkpoint_dir}")
        return safetensors_load_file(single_path)

    with open(index_path, "r") as f:
        index_data = json.load(f)
    shard_files = sorted(set(index_data["weight_map"].values()))

    state_dict: Dict[str, torch.Tensor] = {}
    for shard_name in shard_files:
        shard_path = os.path.join(checkpoint_dir, shard_name)
        state_dict.update(safetensors_load_file(shard_path))
    return state_dict


def load_moe_checkpoint_flexible(model: nn.Module, checkpoint_dir: str, strict: bool = True):
    """Load checkpoint from directory with prefix-tolerant key handling."""
    state_dict = load_moe_checkpoint_state_dict(checkpoint_dir)
    if hasattr(model, "load_moe_state_dict_flexible"):
        return model.load_moe_state_dict_flexible(state_dict, strict=strict)
    return model.load_state_dict(state_dict, strict=strict)


if __name__ == "__main__":
    layer = RankMoELoRALayer(d_in=4096, d_out=4096, r_max=32, top_k=4, router_hidden_dim=128)
    x = torch.randn(2, 128, 4096)
    y, aux = layer(x, return_aux=True)
    l_task = y.mean()
    l_total = l_task + 1e-4 * aux["mask_l1"] + 1e-2 * aux["rank_sparsity"]
    l_total.backward()
