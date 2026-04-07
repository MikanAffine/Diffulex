from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from diffulex.layer.activation import SiluAndMul
from diffulex.layer.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear, divide
from diffulex.moe.config import (
    get_moe_intermediate_size,
    get_num_experts,
    get_num_experts_per_tok,
    get_norm_topk_prob,
)
from diffulex.moe.dispatcher import build_dispatcher
from diffulex.moe.runner import build_runner
from diffulex.moe.topk import build_topk_router
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight


class FusedMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        *,
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(f"Only silu experts are supported right now, got {hidden_act!r}.")

        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.local_intermediate_size = divide(intermediate_size, self.tp_size)

        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.w13 = nn.Parameter(
            torch.empty(num_experts, self.local_intermediate_size * 2, hidden_size)
        )
        self.w13.weight_loader = self.w13_weight_loader
        self.w2 = nn.Parameter(
            torch.empty(num_experts, hidden_size, self.local_intermediate_size)
        )
        self.w2.weight_loader = self.w2_weight_loader
        self.router = build_topk_router("trivial", top_k=top_k, renormalize=norm_topk_prob, scoring_func="softmax")
        self.dispatcher = build_dispatcher("trivial")
        self.runner = build_runner(
            "trivial",
            num_experts=num_experts,
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            top_k=top_k,
            hidden_act=hidden_act,
            get_w13=lambda: self.w13,
            get_w2=lambda: self.w2,
        )
    
    @classmethod
    def from_config(cls, config) -> "FusedMoE":
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=get_moe_intermediate_size(config),
            num_experts=get_num_experts(config),
            top_k=get_num_experts_per_tok(config),
            hidden_act=getattr(config, "hidden_act", "silu"),
            norm_topk_prob=get_norm_topk_prob(config),
        )

    def w13_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = self.local_intermediate_size
        start_idx = self.tp_rank * shard_size
        gate_weight = loaded_weight[:, : self.intermediate_size].narrow(1, start_idx, shard_size)
        up_weight = loaded_weight[:, self.intermediate_size :].narrow(1, start_idx, shard_size)
        param.data[:, :shard_size].copy_(gate_weight)
        param.data[:, shard_size:].copy_(up_weight)

    def w2_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(2)
        start_idx = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(2, start_idx, shard_size))

    def _load_w13_expert_weight(
        self,
        loaded_weight: torch.Tensor,
        *,
        expert_idx: int,
        shard_id: str,
    ) -> None:
        shard_size = self.local_intermediate_size
        start_idx = self.tp_rank * shard_size
        local_weight = loaded_weight.narrow(0, start_idx, shard_size)
        local_offset = 0 if shard_id == "gate_proj" else shard_size
        self.w13.data[expert_idx, local_offset : local_offset + shard_size].copy_(local_weight)

    def _load_w2_expert_weight(
        self,
        loaded_weight: torch.Tensor,
        *,
        expert_idx: int,
    ) -> None:
        shard_size = self.local_intermediate_size
        start_idx = self.tp_rank * shard_size
        local_weight = loaded_weight.narrow(1, start_idx, shard_size)
        self.w2.data[expert_idx].copy_(local_weight)

    def resolve_checkpoint_weight(self, suffix: str, ctx: LoadContext) -> ResolvedWeight | None:
        if suffix == "gate.weight":
            return ResolvedWeight(param=self.gate.weight)
        if suffix == "w13":
            return ResolvedWeight(param=self.w13)
        if suffix == "w2":
            return ResolvedWeight(param=self.w2)

        match = re.fullmatch(r"experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", suffix)
        if match is None:
            return None

        expert_idx = int(match.group(1))
        proj_name = match.group(2)
        if proj_name in ("gate_proj", "up_proj"):
            return ResolvedWeight(
                loader=lambda loaded_weight, expert_idx=expert_idx, proj_name=proj_name: self._load_w13_expert_weight(
                    loaded_weight,
                    expert_idx=expert_idx,
                    shard_id=proj_name,
                )
            )
        return ResolvedWeight(
            loader=lambda loaded_weight, expert_idx=expert_idx: self._load_w2_expert_weight(
                loaded_weight,
                expert_idx=expert_idx,
            )
        )
    
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, original_shape[-1])

        router_logits = self.gate(flat_hidden_states)
        topk_output = self.router(router_logits)
        dispatch_output = self.dispatcher.dispatch(flat_hidden_states, topk_output)
        combine_input = self.runner(dispatch_output)
        final_hidden_states = self.dispatcher.combine(combine_input)

        return final_hidden_states.reshape(original_shape), router_logits


class ExpertMLP(nn.Module):
    """Dense expert used inside sparse MoE blocks."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(f"Only silu experts are supported right now, got {hidden_act!r}.")

        self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden_states = self.act_fn(torch.cat([gate, up], dim=-1))
        return self.down_proj(hidden_states)


class SparseMoEBlock(nn.Module):
    """Naive sparse MoE block matching SDAR-MoE checkpoint structure."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        *,
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                ExpertMLP(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    hidden_act=hidden_act,
                )
                for _ in range(num_experts)
            ]
        )
        self.router = build_topk_router("trivial", top_k=top_k, renormalize=norm_topk_prob, scoring_func="softmax")
        self.dispatcher = build_dispatcher("trivial")
        self.runner = build_runner("trivial")

    @classmethod
    def from_config(cls, config) -> "SparseMoEBlock":
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=get_moe_intermediate_size(config),
            num_experts=get_num_experts(config),
            top_k=get_num_experts_per_tok(config),
            hidden_act=getattr(config, "hidden_act", "silu"),
            norm_topk_prob=get_norm_topk_prob(config),
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        hidden_size = original_shape[-1]
        flat_hidden_states = hidden_states.reshape(-1, hidden_size)

        router_logits = self.gate(flat_hidden_states)
        topk_output = self.router(router_logits)
        if topk_output.ids is None or topk_output.weights is None:
            raise RuntimeError("SparseMoEBlock requires a top-k router that materializes ids and weights.")

        final_hidden_states = torch.zeros_like(flat_hidden_states)
        expert_mask = F.one_hot(topk_output.ids, num_classes=self.num_experts).permute(2, 1, 0)
        routing_weights = topk_output.weights.to(flat_hidden_states.dtype)

        for expert_idx in range(self.num_experts):
            topk_slot_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue

            expert_input = flat_hidden_states[token_idx]
            expert_output = self.experts[expert_idx](expert_input)
            expert_output = expert_output * routing_weights[token_idx, topk_slot_idx].unsqueeze(-1)
            final_hidden_states.index_add_(0, token_idx, expert_output)

        return final_hidden_states.reshape(original_shape), router_logits


__all__ = ["FusedMoE", "ExpertMLP", "SparseMoEBlock"]
