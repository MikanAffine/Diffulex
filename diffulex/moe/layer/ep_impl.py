import re

import torch
import torch.nn as nn

from diffulex.layer.linear import ReplicatedLinear, divide
from diffulex.moe.layer.base import FusedMoE
from diffulex.moe.layer.a2a import build_a2a_backend
from diffulex.moe.config import (
    get_moe_intermediate_size,
    get_num_experts,
    get_num_experts_per_tok,
    get_norm_topk_prob,
)
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight
from diffulex.utils.parallelism import (
    get_ep_local_expert_bounds,
    get_ep_rank,
    get_ep_world_size,
    global_to_local_expert_id,
    owns_global_expert,
)
from diffulex_kernel import fused_moe, fused_topk

class EPFusedMoE(FusedMoE):
    """
    EP MoE with pluggable communication backend.

    Gate/router and expert GEMM stay in this module.
    The backend only handles token dispatch and output combine.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        *,
        hidden_act: str = "silu",
        norm_topk_prob: bool = True,
        a2a_backend: str = "none",
    ) -> None:
        super().__init__(
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            hidden_act=hidden_act,
            norm_topk_prob=norm_topk_prob
        )
        
        self.ep_rank = get_ep_rank()
        self.ep_size = get_ep_world_size()
        self.num_local_experts = divide(self.num_experts, self.ep_size)
        self.local_expert_start, _ = get_ep_local_expert_bounds(self.num_experts)

        # Every rank owns 1 / ep_size of experts.
        self.gate = ReplicatedLinear(hidden_size, self.num_experts, bias=False)
        self.w13 = nn.Parameter(
            torch.empty(self.num_local_experts, self.intermediate_size * 2, hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, self.intermediate_size)
        )
        self.a2a_backend = build_a2a_backend(a2a_backend, self)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, original_shape[-1])

        router_logits = self.gate(flat_hidden_states)
        topk_weights, topk_ids = fused_topk(
            router_logits=router_logits,
            top_k=self.topk,
            renormalize=self.topk_renormalize,
            scoring_func=self.topk_scoring_func,
        )

        dispatch_state = self.a2a_backend.dispatch(
            flat_hidden_states,
            topk_ids,
            topk_weights,
        )
        local_outputs = fused_moe(
            hidden_states=dispatch_state.hidden_states,
            w13=self.w13,
            w2=self.w2,
            topk_ids=dispatch_state.topk_ids,
            topk_weights=dispatch_state.topk_weights,
            local_expert_start=dispatch_state.local_expert_start,
            hidden_act=self.hidden_act,
            topk_ids_are_local=dispatch_state.topk_ids_are_local,
        )
        final_hidden_states = self.a2a_backend.combine(local_outputs, dispatch_state)

        return final_hidden_states.reshape(original_shape), router_logits

    @classmethod
    def from_config(cls, config) -> "EPFusedMoE":
        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=get_moe_intermediate_size(config),
            num_experts=get_num_experts(config),
            top_k=get_num_experts_per_tok(config),
            hidden_act=getattr(config, "hidden_act", "silu"),
            norm_topk_prob=get_norm_topk_prob(config),
        )
    
    def load_w1(self, loaded_weight: torch.Tensor, local_expert_idx: int) -> None:
        self.w13.data[local_expert_idx, 0 : self.intermediate_size].copy_(loaded_weight)

    def load_w3(self, loaded_weight: torch.Tensor, local_expert_idx: int) -> None:
        self.w13.data[local_expert_idx, self.intermediate_size : 2 * self.intermediate_size].copy_(loaded_weight)

    def load_w2(self, loaded_weight: torch.Tensor, local_expert_idx: int) -> None:
        self.w2.data[local_expert_idx].copy_(loaded_weight)

    def resolve_checkpoint_weight(self, suffix: str, ctx: LoadContext) -> ResolvedWeight | None:
        match = re.fullmatch(r"experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", suffix)
        if match is None:
            return None

        expert_idx = int(match.group(1))
        if not owns_global_expert(self.num_experts, expert_idx):
            return ResolvedWeight(skip=True)

        local_expert_idx = global_to_local_expert_id(self.num_experts, expert_idx)
        proj_name = match.group(2)

        if proj_name == "gate_proj":
            return ResolvedWeight(
                loader=lambda loaded_weight, local_expert_idx=local_expert_idx: self.load_w1(
                    loaded_weight,
                    local_expert_idx,
                )
            )

        if proj_name == "up_proj":
            return ResolvedWeight(
                loader=lambda loaded_weight, local_expert_idx=local_expert_idx: self.load_w3(
                    loaded_weight,
                    local_expert_idx,
                )
            )

        if proj_name == "down_proj":
            return ResolvedWeight(
                loader=lambda loaded_weight, local_expert_idx=local_expert_idx: self.load_w2(
                    loaded_weight,
                    local_expert_idx,
                )
            )

        return None


__all__ = ["EPFusedMoE"]
