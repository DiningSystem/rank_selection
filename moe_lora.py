import math
import types
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class LoRAExpert(nn.Module):
    """Single fixed-rank LoRA expert with learnable sigmoid masks.

    Expert math:
        ΔW = (B ⊙ sigmoid(S_b)) @ (A ⊙ sigmoid(S_a))
    where B in R^{d_out x r}, A in R^{r x d_in}.
    """

    def __init__(self, d_in: int, d_out: int, rank: int) -> None:
        super().__init__()
        if rank <= 0 or (rank & (rank - 1)) != 0:
            raise ValueError(f"rank must be a positive power of 2, got {rank}")

        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank

        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.empty(d_out, rank))

        # Learnable mask logits; mask = sigmoid(S)
        self.S_a = nn.Parameter(torch.zeros(rank, d_in))
        self.S_b = nn.Parameter(torch.zeros(d_out, rank))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def masked_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        m_a = torch.sigmoid(self.S_a)
        m_b = torch.sigmoid(self.S_b)
        return self.A * m_a, self.B * m_b

    def rank1_components(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns masked vectors used in rank-1 decomposition.

        If B_tilde[:, i] is b_i~ and A_tilde[i, :] is a_i~, then:
            ΔW = sum_i b_i~ @ (a_i~)^T
        """
        return self.masked_factors()

    def delta_weight(self) -> torch.Tensor:
        a_tilde, b_tilde = self.rank1_components()
        # Equivalent to sum_i (b_i~)(a_i~)^T, implemented without explicit loops.
        return torch.einsum("oi,ij->oj", b_tilde, a_tilde)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies only the LoRA residual: x -> x @ ΔW^T."""
        a_tilde, b_tilde = self.masked_factors()
        hidden = F.linear(x, a_tilde)   # [..., rank]
        return F.linear(hidden, b_tilde)  # [..., d_out]

    def mask_l1(self) -> torch.Tensor:
        # abs(sigmoid(.)) == sigmoid(.) but kept explicit for readability
        return torch.sigmoid(self.S_a).abs().sum() + torch.sigmoid(self.S_b).abs().sum()

    def mask_mean(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.sigmoid(self.S_a).mean(), torch.sigmoid(self.S_b).mean()


class Router(nn.Module):
    """Token router that outputs expert probabilities π(x)."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        hidden_dim: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        if hidden_dim is None:
            self.net = nn.Linear(d_model, num_experts, bias=bias)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim, bias=bias),
                nn.GELU(),
                nn.Linear(hidden_dim, num_experts, bias=bias),
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class MoELoRALayer(nn.Module):
    """Linear layer + routed mixture of masked LoRA experts.

    y = W x + sum_k π_k(x) * (ΔW_k x)
    with optional top-k routing for efficiency.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        experts_config: Sequence[Dict[str, int]],
        top_k: int = 1,
        router_hidden_dim: Optional[int] = None,
        bias: bool = True,
        freeze_base: bool = True,
        expert_chunk_size: int = 0,
        gradient_checkpoint_experts: bool = False,
    ) -> None:
        super().__init__()
        if len(experts_config) == 0:
            raise ValueError("experts_config cannot be empty")
        if top_k <= 0:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        self.d_in = d_in
        self.d_out = d_out
        self.num_experts = len(experts_config)
        self.top_k = min(top_k, self.num_experts)
        self.expert_chunk_size = expert_chunk_size
        self.gradient_checkpoint_experts = gradient_checkpoint_experts

        self.base = nn.Linear(d_in, d_out, bias=bias)
        if freeze_base:
            self.base.weight.requires_grad = False
            if self.base.bias is not None:
                self.base.bias.requires_grad = False

        self.experts = nn.ModuleList(
            [LoRAExpert(d_in=d_in, d_out=d_out, rank=cfg["rank"]) for cfg in experts_config]
        )
        self.router = Router(d_model=d_in, num_experts=self.num_experts, hidden_dim=router_hidden_dim)

        self.register_buffer("expert_usage_counts", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("routed_token_count", torch.tensor(0.0), persistent=False)

        self._last_router_probs: Optional[torch.Tensor] = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        experts_config: Sequence[Dict[str, int]],
        top_k: int = 1,
        router_hidden_dim: Optional[int] = None,
        freeze_base: bool = True,
        expert_chunk_size: int = 0,
        gradient_checkpoint_experts: bool = False,
    ) -> "MoELoRALayer":
        layer = cls(
            d_in=linear.in_features,
            d_out=linear.out_features,
            experts_config=experts_config,
            top_k=top_k,
            router_hidden_dim=router_hidden_dim,
            bias=linear.bias is not None,
            freeze_base=freeze_base,
            expert_chunk_size=expert_chunk_size,
            gradient_checkpoint_experts=gradient_checkpoint_experts,
        )
        with torch.no_grad():
            layer.base.weight.copy_(linear.weight)
            if linear.bias is not None and layer.base.bias is not None:
                layer.base.bias.copy_(linear.bias)
        return layer

    def _flatten_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        return x_2d, original_shape

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x_2d, original_shape = self._flatten_tokens(x)

        base_out = self.base(x_2d)

        _, probs = self.router(x_2d)
        self._last_router_probs = probs

        top_vals, top_idx = torch.topk(probs, k=self.top_k, dim=-1)

        delta_out = torch.zeros_like(base_out)

        # Compute only experts selected by top-k routing.
        for expert_id, expert in enumerate(self.experts):
            token_mask = (top_idx == expert_id)
            if not token_mask.any():
                continue

            token_positions, slot_positions = token_mask.nonzero(as_tuple=True)
            x_selected = x_2d[token_positions]
            weights = top_vals[token_positions, slot_positions]

            chunk_size = self.expert_chunk_size if self.expert_chunk_size > 0 else x_selected.size(0)
            for start in range(0, x_selected.size(0), chunk_size):
                end = start + chunk_size
                x_chunk = x_selected[start:end]
                token_chunk = token_positions[start:end]
                weight_chunk = weights[start:end]

                if self.training and self.gradient_checkpoint_experts:
                    expert_out = checkpoint.checkpoint(expert, x_chunk, use_reentrant=False)
                else:
                    expert_out = expert(x_chunk)

                weighted = expert_out * weight_chunk.unsqueeze(-1)
                delta_out.index_add_(0, token_chunk, weighted)

            with torch.no_grad():
                self.expert_usage_counts[expert_id] += float(token_positions.numel())

        with torch.no_grad():
            self.routed_token_count += float(x_2d.size(0))

        y = (base_out + delta_out).reshape(*original_shape[:-1], self.d_out)

        if not return_aux:
            return y

        aux = {
            "mask_l1": self.mask_sparsity_loss(),
            "router_balance": self.router_balance_loss(probs),
            "expert_usage": self.expert_usage_frequency(),
            "effective_sparsity": self.effective_sparsity(),
        }
        return y, aux

    def mask_sparsity_loss(self) -> torch.Tensor:
        return sum(expert.mask_l1() for expert in self.experts)

    def router_balance_loss(self, router_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = self._last_router_probs if router_probs is None else router_probs
        if probs is None:
            raise RuntimeError("router_balance_loss requires a prior forward pass or explicit router_probs")

        p = probs.mean(dim=0)
        p = torch.clamp(p, min=1e-9)
        return torch.sum(p * torch.log(p))

    def aux_loss(
        self,
        lambda_mask: float,
        lambda_router: float,
        router_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return lambda_mask * self.mask_sparsity_loss() + lambda_router * self.router_balance_loss(router_probs)

    def expert_usage_frequency(self) -> torch.Tensor:
        denom = torch.clamp(self.routed_token_count, min=1.0)
        return self.expert_usage_counts / denom

    def effective_sparsity(self) -> Dict[str, torch.Tensor]:
        mean_a = []
        mean_b = []
        for expert in self.experts:
            m_a, m_b = expert.mask_mean()
            mean_a.append(m_a)
            mean_b.append(m_b)
        return {
            "mask_a_mean": torch.stack(mean_a),
            "mask_b_mean": torch.stack(mean_b),
        }


@dataclass
class MoELoRAConfig:
    experts_config: Sequence[Dict[str, int]]
    top_k: int = 1
    router_hidden_dim: Optional[int] = None
    target_modules: Optional[Sequence[str]] = None
    freeze_base: bool = True
    expert_chunk_size: int = 0
    gradient_checkpoint_experts: bool = False

    def __post_init__(self) -> None:
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")
        if not self.experts_config:
            raise ValueError("experts_config cannot be empty")


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
        if isinstance(module, MoELoRALayer):
            for param in module.experts.parameters():
                param.requires_grad = True
            for param in module.router.parameters():
                param.requires_grad = True


def merge_and_unload_moe_lora(self):
    raise NotImplementedError(
        "MoE-LoRA uses input-dependent routing and cannot be exactly merged into a single static linear layer."
    )


def apply_moe_lora(model: nn.Module, config: MoELoRAConfig):
    model.peft_config = getattr(model, "peft_config", {})
    model.adapter_layers = getattr(model, "adapter_layers", set())
    model.peft_config["moe_lora"] = config

    target_modules = list(config.target_modules)
    replaced_count = 0
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name, target_name = _get_submodules(model, name)
                parent = model
                for name_part in parent_name.split("."):
                    if name_part:
                        parent = getattr(parent, name_part)

                wrapped = MoELoRALayer.from_linear(
                    module,
                    experts_config=config.experts_config,
                    top_k=config.top_k,
                    router_hidden_dim=config.router_hidden_dim,
                    freeze_base=config.freeze_base,
                    expert_chunk_size=config.expert_chunk_size,
                    gradient_checkpoint_experts=config.gradient_checkpoint_experts,
                )
                setattr(parent, target_name, wrapped)
                model.adapter_layers.add(name)
                replaced_count += 1

    if replaced_count == 0:
        raise RuntimeError(
            "No target nn.Linear modules were replaced by MoE-LoRA. "
            "Check `target_modules` against `model.named_modules()`."
        )

    model.mark_only_adapters_as_trainable = types.MethodType(mark_only_moe_lora_as_trainable, model)
    model.merge_and_unload = types.MethodType(merge_and_unload_moe_lora, model)
    mark_only_moe_lora_as_trainable(model)
    return model


def get_moe_lora_model(model: nn.Module, config: MoELoRAConfig):
    return apply_moe_lora(model, config)


if __name__ == "__main__":
    # Example: replacing nn.Linear with MoE-LoRA.
    experts_config = [
        {"rank": 4},
        {"rank": 8},
        {"rank": 16},
        {"rank": 32},
    ]

    layer = MoELoRALayer(d_in=4096, d_out=4096, experts_config=experts_config, top_k=2)
    x = torch.randn(2, 128, 4096)
    y, aux = layer(x, return_aux=True)

    l_task = y.mean()
    l_total = l_task + 1e-4 * aux["mask_l1"] + 1e-2 * aux["router_balance"]
    l_total.backward()
