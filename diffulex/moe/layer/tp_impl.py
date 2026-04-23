import os
import re
import time

import torch
import torch.nn as nn
import torch.distributed as dist

from diffulex.layer.linear import ReplicatedLinear, divide, tp_all_reduce
from diffulex.moe.layer.base import FusedMoE
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight
from diffulex.distributed.parallel_state import fetch_parallel_state
from diffulex.logger import get_logger

logger = get_logger(__name__)

TP_FP32_ALLREDUCE_MOE = os.environ.get("DIFFULEX_TP_FP32_ALLREDUCE", "0") == "1" or os.environ.get(
    "DIFFULEX_TP_FP32_ALLREDUCE_MOE", "0"
) == "1"


class TPFusedMoE(FusedMoE):
    """
    Standard TP MoE without token dispatch.

    Every TP rank receives the full local token batch and owns a contiguous
    shard of experts. The router still selects global expert ids; each rank
    computes only the selected experts it owns, then TP all-reduce sums the
    partial expert contributions.
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
        num_shared_experts: int = 0,
        shared_expert_intermediate_size: int | None = None,
    ) -> None:
        super().__init__(
            hidden_size,
            intermediate_size,
            num_experts,
            top_k,
            hidden_act=hidden_act,
            norm_topk_prob=norm_topk_prob,
            num_shared_experts=num_shared_experts,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
        )
        
        parallel_state = fetch_parallel_state()
        self.tp_rank = parallel_state.get_tp_rank()
        self.tp_size = parallel_state.get_tp_world_size()
        self.tp_group = parallel_state.get_tp_group()
        self.num_local_experts = divide(self.num_experts, self.tp_size)
        self.local_expert_start = self.tp_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts

        self.gate = ReplicatedLinear(hidden_size, self.num_experts, bias=False)
        self.w13 = nn.Parameter(
            torch.empty(self.num_local_experts, self.intermediate_size * 2, hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, self.intermediate_size)
        )
        self._moe_stage_timing = os.getenv("DIFFULEX_MOE_STAGE_TIMING", "0") == "1"
        self._moe_stage_timing_sync = os.getenv("DIFFULEX_MOE_STAGE_TIMING_SYNC", "0") == "1"
        self._moe_stage_timing_interval = max(1, int(os.getenv("DIFFULEX_MOE_STAGE_TIMING_INTERVAL", "20")))
        self._moe_stage_timing_steps = 0
        self._moe_stage_time_acc = {
            "gate": 0.0,
            "router": 0.0,
            "gemm": 0.0,
            "all_reduce": 0.0,
            "shared": 0.0,
        }

    def _mark_time(self) -> float:
        if self._moe_stage_timing_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _report_stage_timing(self) -> None:
        if not self._moe_stage_timing:
            return
        self._moe_stage_timing_steps += 1
        if self._moe_stage_timing_steps % self._moe_stage_timing_interval != 0:
            return
        denom = float(self._moe_stage_timing_interval)
        avg = {k: (v / denom) * 1000.0 for k, v in self._moe_stage_time_acc.items()}
        self._moe_stage_time_acc = {k: 0.0 for k in self._moe_stage_time_acc}
        if self.tp_rank == 0:
            logger.info(
                "[moe-timing] interval=%d avg_ms gate=%.3f router=%.3f gemm=%.3f all_reduce=%.3f shared=%.3f",
                self._moe_stage_timing_interval,
                avg["gate"],
                avg["router"],
                avg["gemm"],
                avg["all_reduce"],
                avg["shared"],
            )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, original_shape[-1])

        t0 = self._mark_time() if self._moe_stage_timing else 0.0
        router_logits = self.gate(flat_hidden_states)
        t1 = self._mark_time() if self._moe_stage_timing else 0.0
        topk_output = self.router(router_logits)
        t2 = self._mark_time() if self._moe_stage_timing else 0.0
        topk_weights = topk_output.weights
        topk_ids = topk_output.ids
        try:
            from diffulex.model.llada2 import _llada2_debug_enabled, _llada2_debug_store_nested
        except Exception:
            _llada2_debug_enabled = None
            _llada2_debug_store_nested = None
        if _llada2_debug_enabled is not None and _llada2_debug_enabled():
            layer_idx = getattr(self, "_llada2_layer_idx", None)
            if layer_idx is not None:
                _llada2_debug_store_nested("moe", f"layer_{layer_idx}_gate_input", flat_hidden_states)
                _llada2_debug_store_nested("moe", f"layer_{layer_idx}_topk_ids", topk_ids)
                _llada2_debug_store_nested("moe", f"layer_{layer_idx}_topk_weights", topk_weights)
                if hasattr(self.gate, "expert_bias"):
                    _llada2_debug_store_nested("moe", f"layer_{layer_idx}_expert_bias", self.gate.expert_bias)
        final_hidden_states = self.expert_gemm(
            impl="triton",
            hidden_states=flat_hidden_states,
            w13=self.w13,
            w2=self.w2,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            local_expert_start=self.local_expert_start,
            hidden_act=self.hidden_act,
        )
        t3 = self._mark_time() if self._moe_stage_timing else 0.0
        if self.tp_size > 1:
            final_hidden_states = tp_all_reduce(
                final_hidden_states,
                self.tp_group,
                fp32=TP_FP32_ALLREDUCE_MOE,
            )
        t4 = self._mark_time() if self._moe_stage_timing else 0.0

        final_hidden_states = final_hidden_states.reshape(original_shape)
        final_hidden_states = self.add_shared_experts(final_hidden_states, hidden_states)
        t5 = self._mark_time() if self._moe_stage_timing else 0.0
        if self._moe_stage_timing:
            self._moe_stage_time_acc["gate"] += max(0.0, t1 - t0)
            self._moe_stage_time_acc["router"] += max(0.0, t2 - t1)
            self._moe_stage_time_acc["gemm"] += max(0.0, t3 - t2)
            self._moe_stage_time_acc["all_reduce"] += max(0.0, t4 - t3)
            self._moe_stage_time_acc["shared"] += max(0.0, t5 - t4)
            self._report_stage_timing()
        return final_hidden_states, router_logits

    def _local_expert_idx(self, expert_idx: int) -> int | None:
        if expert_idx < self.local_expert_start or expert_idx >= self.local_expert_end:
            return None
        return expert_idx - self.local_expert_start
    
    def load_w1(self, loaded_weight: torch.Tensor, expert_idx: int) -> None:
        local_expert_idx = self._local_expert_idx(expert_idx)
        if local_expert_idx is None:
            return
        # loaded_weight: [intermediate_size, hidden_size]
        self.w13.data[local_expert_idx, 0:self.intermediate_size].copy_(loaded_weight)
    
    def load_w3(self, loaded_weight: torch.Tensor, expert_idx: int) -> None:
        local_expert_idx = self._local_expert_idx(expert_idx)
        if local_expert_idx is None:
            return
        # loaded_weight: [intermediate_size, hidden_size]
        self.w13.data[local_expert_idx, self.intermediate_size:2*self.intermediate_size].copy_(loaded_weight)
    
    def load_w2(self, loaded_weight: torch.Tensor, expert_idx: int) -> None:
        local_expert_idx = self._local_expert_idx(expert_idx)
        if local_expert_idx is None:
            return
        target = self.w2.data[local_expert_idx]  # [hidden_size, intermediate_size]
        if loaded_weight.shape == (self.hidden_size, self.intermediate_size):
            shard = loaded_weight
        elif loaded_weight.shape == (self.intermediate_size, self.hidden_size):
            shard = loaded_weight.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                f"Unexpected down_proj weight shape: {loaded_weight.shape}, "
                f"target shape: {target.shape}"
            )
        target.copy_(shard)

    def resolve_checkpoint_weight(self, suffix: str, ctx: LoadContext) -> ResolvedWeight | None:
        match = re.fullmatch(r"experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", suffix)
        if match is None:
            return None

        expert_idx = int(match.group(1))
        proj_name = match.group(2)

        if proj_name == "gate_proj":
            return ResolvedWeight(
                loader=lambda loaded_weight, expert_idx=expert_idx: self.load_w1(
                    loaded_weight,
                    expert_idx,
                )
            )

        if proj_name == "up_proj":
            return ResolvedWeight(
                loader=lambda loaded_weight, expert_idx=expert_idx: self.load_w3(
                    loaded_weight,
                    expert_idx,
                )
            )
        
        if proj_name == "down_proj":
            return ResolvedWeight(
                loader=lambda loaded_weight, expert_idx=expert_idx: self.load_w2(
                    loaded_weight,
                    expert_idx,
                )
            )
        
        return None

__all__ = ["TPFusedMoE"]
