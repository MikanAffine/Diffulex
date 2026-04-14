import re

import torch
import torch.nn as nn
import torch.distributed as dist

from diffulex.layer.linear import ReplicatedLinear, divide
from diffulex.moe.layer.base import FusedMoE
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight
from diffulex.utils.parallelism import get_ep_rank, get_ep_world_size
from diffulex_kernel import fused_moe, fused_topk

class EPFusedMoE(FusedMoE):
    """
    if ep is on, moe layer will only use ep even if tp is on
    so whole expert weight is distributed to ep_size devices

    have all-to-all token dispatch, each rank computes 1/ep_size
    of gate and topk of tokens, and send to owner, then combine
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
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.active_expert_ids = list(range(self.local_expert_start, self.local_expert_end))

        # every rank process 1 / ep_size of total tokens and do a2a communication
        self.gate = ReplicatedLinear(hidden_size, self.num_experts, bias=False)
        self.w13 = nn.Parameter(
            torch.empty(self.num_local_experts, self.intermediate_size * 2, hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, self.intermediate_size)
        )

    def shard_tokens(self, flat_hidden_states):
        num_tokens = flat_hidden_states.shape[0]
        local_token_indices = torch.arange(
            self.ep_rank,
            num_tokens,
            self.ep_size,
            device=flat_hidden_states.device,
            dtype=torch.int32,
        )
        local_hidden_states = flat_hidden_states.index_select(0, local_token_indices.to(torch.int64))
        return local_hidden_states, local_token_indices, num_tokens

    @staticmethod
    def _all_to_all_tensor(
        send_tensor: torch.Tensor,
        *,
        send_splits: list[int],
        recv_splits: list[int],
    ) -> torch.Tensor:
        recv_shape = (sum(recv_splits), *send_tensor.shape[1:])
        recv_tensor = torch.empty(
            recv_shape,
            device=send_tensor.device,
            dtype=send_tensor.dtype,
        )
        dist.all_to_all_single(
            recv_tensor,
            send_tensor.contiguous(),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        return recv_tensor

    def _num_tokens_for_rank(self, num_tokens: int, rank: int) -> int:
        remaining = num_tokens - rank
        if remaining <= 0:
            return 0
        return (remaining + self.ep_size - 1) // self.ep_size

    def _max_local_tokens(self, num_tokens: int) -> int:
        return (num_tokens + self.ep_size - 1) // self.ep_size

    def _gather_local_outputs(
        self,
        local_final_hidden_states: torch.Tensor,
        *,
        num_tokens: int,
    ) -> torch.Tensor:
        if self.ep_size == 1:
            return local_final_hidden_states

        num_local_tokens, hidden_size = local_final_hidden_states.shape
        max_local_tokens = self._max_local_tokens(num_tokens)
        padded_local_outputs = torch.zeros(
            (max_local_tokens, hidden_size),
            device=local_final_hidden_states.device,
            dtype=local_final_hidden_states.dtype,
        )
        padded_local_outputs[:num_local_tokens] = local_final_hidden_states.contiguous()

        gathered_outputs = [
            torch.empty_like(padded_local_outputs)
            for _ in range(self.ep_size)
        ]
        dist.all_gather(gathered_outputs, padded_local_outputs)

        final_hidden_states = torch.empty(
            (num_tokens, hidden_size),
            device=local_final_hidden_states.device,
            dtype=local_final_hidden_states.dtype,
        )
        for rank in range(self.ep_size):
            local_count = self._num_tokens_for_rank(num_tokens, rank)
            if local_count == 0:
                continue
            final_hidden_states[rank::self.ep_size] = gathered_outputs[rank][:local_count]
        return final_hidden_states

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phase = self.get_current_phase()
        if phase == "prefill":
            return self._forward_replicated(hidden_states)
        if phase == "decode":
            return self._forward_replicated(hidden_states)
        return self._forward_replicated(hidden_states)

    def _forward_replicated(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, original_shape[-1])

        router_logits = self.gate(flat_hidden_states)
        topk_weights, topk_ids = fused_topk(
            router_logits=router_logits,
            top_k=self.topk,
            renormalize=self.topk_renormalize,
            scoring_func=self.topk_scoring_func,
        )

        final_hidden_states = fused_moe(
            hidden_states=flat_hidden_states,
            w13=self.w13,
            w2=self.w2,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            local_expert_start=self.local_expert_start,
            hidden_act=self.hidden_act,
        )
        dist.all_reduce(final_hidden_states)

        return final_hidden_states.reshape(original_shape), router_logits

    def _a2a_dispatch_tokens(
        self,
        local_hidden_states: torch.Tensor,
        topk_ids_global: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        num_tokens: int,
    ) -> dict[str, torch.Tensor | list[int] | int]:
        device = local_hidden_states.device
        dtype = local_hidden_states.dtype
        hidden_size = local_hidden_states.shape[1]
        ep_size = self.ep_size
        top_k = topk_ids_global.shape[1]
        num_local_tokens = local_hidden_states.shape[0]

        slot_token_positions = (
            torch.arange(num_local_tokens, device=device, dtype=torch.int32)
            .unsqueeze(1)
            .expand(num_local_tokens, top_k)
            .reshape(-1)
            .contiguous()
        )
        slot_expert_ids_global = topk_ids_global.reshape(-1).contiguous()
        slot_weights = topk_weights.reshape(-1).contiguous()

        dst_rank = torch.div(
            slot_expert_ids_global,
            self.num_local_experts,
            rounding_mode="floor",
        ).to(torch.int32)
        dst_local_expert = (
            slot_expert_ids_global - dst_rank * self.num_local_experts
        ).to(torch.int32).contiguous()

        sort_idx = torch.argsort(dst_rank)
        dst_rank_sorted = dst_rank[sort_idx]
        source_token_positions = slot_token_positions[sort_idx].contiguous()
        hidden_token_positions = torch.div(sort_idx, top_k, rounding_mode="floor").to(torch.int64)
        send_hidden_states = local_hidden_states.index_select(0, hidden_token_positions).contiguous()
        send_local_expert = dst_local_expert[sort_idx].contiguous()
        send_weights = slot_weights[sort_idx].contiguous()

        send_counts = torch.bincount(dst_rank_sorted.to(torch.int64), minlength=ep_size).to(torch.int32)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)

        send_splits = send_counts.cpu().tolist()
        recv_splits = recv_counts.cpu().tolist()
        recv_hidden_states = self._all_to_all_tensor(
            send_hidden_states,
            send_splits=send_splits,
            recv_splits=recv_splits,
        )
        recv_local_expert = self._all_to_all_tensor(
            send_local_expert,
            send_splits=send_splits,
            recv_splits=recv_splits,
        )
        recv_weights = self._all_to_all_tensor(
            send_weights,
            send_splits=send_splits,
            recv_splits=recv_splits,
        )

        return dict(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
            num_local_tokens=num_local_tokens,
            send_splits=send_splits,
            recv_splits=recv_splits,
            source_token_positions=source_token_positions,
            recv_hidden_states=recv_hidden_states,
            recv_local_expert=recv_local_expert,
            recv_weights=recv_weights,
        )

    def _a2a_combine_tokens(
        self,
        recv_slot_outputs: torch.Tensor,
        dispatch_ctx: dict[str, torch.Tensor | list[int] | int],
    ) -> torch.Tensor:
        device = dispatch_ctx["device"]
        dtype = dispatch_ctx["dtype"]
        num_tokens = int(dispatch_ctx["num_tokens"])
        num_local_tokens = int(dispatch_ctx["num_local_tokens"])
        send_splits = dispatch_ctx["send_splits"]
        recv_splits = dispatch_ctx["recv_splits"]
        source_token_positions = dispatch_ctx["source_token_positions"]

        returned_outputs = self._all_to_all_tensor(
            recv_slot_outputs,
            send_splits=recv_splits,
            recv_splits=send_splits,
        )

        local_final_hidden_states = torch.zeros(
            (num_local_tokens, recv_slot_outputs.shape[1]),
            device=device,
            dtype=torch.float32,
        )

        if returned_outputs.shape[0] > 0:
            local_final_hidden_states.index_add_(0, source_token_positions.to(torch.int64), returned_outputs.float())

        local_final_hidden_states = local_final_hidden_states.to(dtype).contiguous()
        return self._gather_local_outputs(local_final_hidden_states, num_tokens=num_tokens)

    def _forward_a2a(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, original_shape[-1])

        num_tokens = flat_hidden_states.shape[0]

        if num_tokens == 0:
            return flat_hidden_states.reshape(original_shape), None

        local_hidden_states, _local_token_indices, _ = self.shard_tokens(flat_hidden_states)
        if local_hidden_states.shape[0] == 0:
            effective_topk = min(self.top_k, self.num_experts)
            topk_weights = torch.empty(
                (0, effective_topk),
                device=flat_hidden_states.device,
                dtype=torch.float32,
            )
            topk_ids_global = torch.empty(
                (0, effective_topk),
                device=flat_hidden_states.device,
                dtype=torch.int32,
            )
        else:
            router_logits_local = self.gate(local_hidden_states)  # [T_local, num_experts]
            topk_weights, topk_ids_global = fused_topk(
                router_logits=router_logits_local,
                top_k=self.topk,
                renormalize=self.topk_renormalize,
                scoring_func=self.topk_scoring_func,
            )

        dispatch_ctx = self._a2a_dispatch_tokens(
            local_hidden_states=local_hidden_states,
            topk_ids_global=topk_ids_global,
            topk_weights=topk_weights,
            num_tokens=num_tokens,
        )

        recv_hidden_states = dispatch_ctx["recv_hidden_states"]
        recv_local_expert = dispatch_ctx["recv_local_expert"]
        if recv_hidden_states.shape[0] > 0:
            recv_topk_ids_local = recv_local_expert[:, None].contiguous()  # [R, 1]
            recv_topk_weights_local = torch.ones(
                (recv_hidden_states.shape[0], 1),
                device=recv_hidden_states.device,
                dtype=topk_weights.dtype,
            )
            recv_slot_outputs = fused_moe(
                hidden_states=recv_hidden_states,
                w13=self.w13,
                w2=self.w2,
                topk_ids=recv_topk_ids_local,
                topk_weights=recv_topk_weights_local,
                local_expert_start=0,
                hidden_act=self.hidden_act,
                topk_ids_are_local=True,
            ).contiguous()  # [R, H]
            recv_slot_outputs.mul_(dispatch_ctx["recv_weights"].to(recv_slot_outputs.dtype).unsqueeze(-1))
        else:
            recv_slot_outputs = torch.empty(
                (0, int(dispatch_ctx["hidden_size"])),
                device=dispatch_ctx["device"],
                dtype=dispatch_ctx["dtype"],
            )

        final_hidden_states = self._a2a_combine_tokens(recv_slot_outputs, dispatch_ctx)
        return final_hidden_states.reshape(original_shape), None
    
    def owns_global_expert(self, expert_idx: int) -> bool:
        return self.local_expert_start <= expert_idx < self.local_expert_end
    
    def global_to_local_expert_id(self, global_expert_idx: int) -> int:
        assert self.owns_global_expert(global_expert_idx), f"global_expert_idx {global_expert_idx} is not owned by this rank"
        return global_expert_idx - self.local_expert_start

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
        if not self.owns_global_expert(expert_idx):
            return ResolvedWeight(skip=True)

        local_expert_idx = self.global_to_local_expert_id(expert_idx)
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
