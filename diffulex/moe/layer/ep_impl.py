import re

import torch
import torch.nn as nn
import torch.distributed as dist

from diffulex.layer.linear import ReplicatedLinear, divide
from diffulex.moe.layer.base import FusedMoE
from diffulex.utils.checkpoint import LoadContext, ResolvedWeight
from diffulex.utils.parallelism import get_ep_rank, get_ep_world_size


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
        token_indices = torch.arange(num_tokens, device=flat_hidden_states.device)
        local_token_indices = token_indices[self.ep_rank::self.ep_size]
        local_hidden_states = flat_hidden_states[local_token_indices]
        return local_hidden_states, local_token_indices, num_tokens

    @staticmethod
    def _pack_dispatch_metadata(local_expert_ids: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        return (token_indices.to(torch.int64) << 32) | local_expert_ids.to(torch.int64)

    @staticmethod
    def _unpack_dispatch_metadata(metadata: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_indices = torch.bitwise_right_shift(metadata, 32).to(torch.int32)
        local_expert_ids = torch.bitwise_and(metadata, 0xFFFFFFFF).to(torch.int32)
        return local_expert_ids, token_indices

    @staticmethod
    def _pack_dispatch_payload(
        hidden_states: torch.Tensor,
        metadata: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        num_slots, hidden_size = hidden_states.shape
        hidden_bytes = hidden_size * hidden_states.element_size()
        metadata_bytes = metadata.element_size()
        weight_bytes = weights.element_size()
        payload = torch.empty(
            (num_slots, hidden_bytes + metadata_bytes + weight_bytes),
            device=hidden_states.device,
            dtype=torch.uint8,
        )
        payload[:, :hidden_bytes].copy_(
            hidden_states.contiguous().view(torch.uint8).reshape(num_slots, hidden_bytes)
        )
        offset = hidden_bytes
        payload[:, offset : offset + metadata_bytes].copy_(
            metadata.contiguous().view(torch.uint8).reshape(num_slots, metadata_bytes)
        )
        offset += metadata_bytes
        payload[:, offset : offset + weight_bytes].copy_(
            weights.contiguous().view(torch.uint8).reshape(num_slots, weight_bytes)
        )
        return payload

    @staticmethod
    def _unpack_dispatch_payload(
        payload: torch.Tensor,
        *,
        hidden_size: int,
        hidden_dtype: torch.dtype,
        weight_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_slots = payload.shape[0]
        hidden_bytes = hidden_size * torch.empty((), dtype=hidden_dtype).element_size()
        metadata_bytes = torch.empty((), dtype=torch.int64).element_size()
        weight_bytes = torch.empty((), dtype=weight_dtype).element_size()

        hidden_states = payload[:, :hidden_bytes].contiguous().view(hidden_dtype).reshape(num_slots, hidden_size)
        offset = hidden_bytes
        metadata = payload[:, offset : offset + metadata_bytes].contiguous().view(torch.int64).reshape(num_slots)
        offset += metadata_bytes
        weights = payload[:, offset : offset + weight_bytes].contiguous().view(weight_dtype).reshape(num_slots)
        return hidden_states, metadata, weights

    @staticmethod
    def _pack_return_payload(
        outputs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        num_slots, hidden_size = outputs.shape
        hidden_bytes = hidden_size * outputs.element_size()
        token_bytes = token_indices.element_size()
        payload = torch.empty(
            (num_slots, hidden_bytes + token_bytes),
            device=outputs.device,
            dtype=torch.uint8,
        )
        payload[:, :hidden_bytes].copy_(
            outputs.contiguous().view(torch.uint8).reshape(num_slots, hidden_bytes)
        )
        payload[:, hidden_bytes : hidden_bytes + token_bytes].copy_(
            token_indices.contiguous().view(torch.uint8).reshape(num_slots, token_bytes)
        )
        return payload

    @staticmethod
    def _unpack_return_payload(
        payload: torch.Tensor,
        *,
        hidden_size: int,
        hidden_dtype: torch.dtype,
        token_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_slots = payload.shape[0]
        hidden_bytes = hidden_size * torch.empty((), dtype=hidden_dtype).element_size()
        token_bytes = torch.empty((), dtype=token_dtype).element_size()
        outputs = payload[:, :hidden_bytes].contiguous().view(hidden_dtype).reshape(num_slots, hidden_size)
        token_indices = payload[:, hidden_bytes : hidden_bytes + token_bytes].contiguous().view(token_dtype).reshape(
            num_slots
        )
        return outputs, token_indices

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phase = self.get_current_phase()
        if phase == "prefill":
            return self._forward_replicated(hidden_states)
        if phase == "decode":
            #return self._forward_a2a(hidden_states)
            # a2a performance is worse now
            return self._forward_replicated(hidden_states)
        return self._forward_replicated(hidden_states)

    def _forward_replicated(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, original_shape[-1])

        router_logits = self.gate(flat_hidden_states)
        topk_output = self.router(router_logits)
        topk_weights = topk_output.weights
        topk_ids = topk_output.ids

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
        dist.all_reduce(final_hidden_states)

        return final_hidden_states.reshape(original_shape), router_logits

    def _a2a_dispatch_tokens(
        self,
        flat_hidden_states: torch.Tensor,
        topk_ids_global: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[int] | int]:
        device = flat_hidden_states.device
        dtype = flat_hidden_states.dtype

        num_tokens, hidden_size = flat_hidden_states.shape
        ep_rank = self.ep_rank
        ep_size = self.ep_size
        top_k = self.top_k

        global_token_indices = torch.arange(num_tokens, device=device, dtype=torch.int32)
        local_token_indices = global_token_indices[ep_rank::ep_size].contiguous()
        local_hidden_states = flat_hidden_states[local_token_indices].contiguous()
        num_local_tokens = local_hidden_states.shape[0]

        num_slots = num_local_tokens * top_k
        slot_hidden_states = local_hidden_states.repeat_interleave(top_k, dim=0).contiguous()
        slot_token_indices = (
            local_token_indices.unsqueeze(1)
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
        send_hidden_states = slot_hidden_states[sort_idx].contiguous()
        send_local_expert = dst_local_expert[sort_idx].contiguous()
        send_token_indices = slot_token_indices[sort_idx].contiguous()
        send_weights = slot_weights[sort_idx].contiguous()
        send_metadata = self._pack_dispatch_metadata(send_local_expert, send_token_indices).contiguous()
        send_payload = self._pack_dispatch_payload(send_hidden_states, send_metadata, send_weights)

        send_counts = torch.bincount(dst_rank_sorted.to(torch.int64), minlength=ep_size).to(torch.int32)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)

        send_splits = send_counts.cpu().tolist()
        recv_splits = recv_counts.cpu().tolist()
        total_recv_slots = int(recv_counts.sum().item())
        payload_width = send_payload.shape[1]
        recv_payload = torch.empty(
            (total_recv_slots, payload_width),
            device=device,
            dtype=torch.uint8,
        )

        dist.all_to_all_single(
            recv_payload,
            send_payload,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
        )
        recv_hidden_states, recv_metadata, recv_weights = self._unpack_dispatch_payload(
            recv_payload,
            hidden_size=hidden_size,
            hidden_dtype=dtype,
            weight_dtype=topk_weights.dtype,
        )
        recv_local_expert, recv_token_indices = self._unpack_dispatch_metadata(recv_metadata)

        return dict(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
            num_local_tokens=num_local_tokens,
            local_token_indices=local_token_indices,
            send_splits=send_splits,
            recv_splits=recv_splits,
            recv_hidden_states=recv_hidden_states,
            recv_local_expert=recv_local_expert,
            recv_token_indices=recv_token_indices,
            recv_weights=recv_weights,
            total_recv_slots=total_recv_slots,
        )

    def _a2a_combine_tokens(
        self,
        recv_slot_outputs: torch.Tensor,
        dispatch_ctx: dict[str, torch.Tensor | list[int] | int],
    ) -> torch.Tensor:
        device = dispatch_ctx["device"]
        dtype = dispatch_ctx["dtype"]
        hidden_size = int(dispatch_ctx["hidden_size"])
        num_tokens = int(dispatch_ctx["num_tokens"])
        num_local_tokens = int(dispatch_ctx["num_local_tokens"])
        local_token_indices = dispatch_ctx["local_token_indices"]
        send_splits = dispatch_ctx["send_splits"]
        recv_splits = dispatch_ctx["recv_splits"]
        recv_token_indices = dispatch_ctx["recv_token_indices"]
        send_back_splits = recv_splits
        recv_back_splits = send_splits
        total_returned_slots = int(sum(recv_back_splits))
        send_back_payload = self._pack_return_payload(recv_slot_outputs, recv_token_indices)
        returned_payload = torch.empty(
            (total_returned_slots, send_back_payload.shape[1]),
            device=device,
            dtype=torch.uint8,
        )

        dist.all_to_all_single(
            returned_payload,
            send_back_payload,
            output_split_sizes=recv_back_splits,
            input_split_sizes=send_back_splits,
        )
        returned_outputs, returned_token_indices = self._unpack_return_payload(
            returned_payload,
            hidden_size=hidden_size,
            hidden_dtype=dtype,
            token_dtype=torch.int32,
        )

        local_final_hidden_states = torch.zeros(
            (num_local_tokens, hidden_size),
            device=device,
            dtype=torch.float32,
        )

        if total_returned_slots > 0:
            local_positions = torch.div(
                returned_token_indices - self.ep_rank,
                self.ep_size,
                rounding_mode="floor",
            ).to(torch.int64)

            local_final_hidden_states.index_add_(0, local_positions, returned_outputs.float())

        local_final_hidden_states = local_final_hidden_states.to(dtype).contiguous()

        local_token_count = torch.tensor(
            [num_local_tokens],
            device=device,
            dtype=torch.int64,
        )
        gathered_counts = [torch.zeros_like(local_token_count) for _ in range(self.ep_size)]
        dist.all_gather(gathered_counts, local_token_count)
        gathered_counts = [int(x.item()) for x in gathered_counts]

        max_local_tokens = max(gathered_counts)
        padded_local_token_indices = torch.full(
            (max_local_tokens,),
            fill_value=-1,
            device=device,
            dtype=torch.int32,
        )
        padded_local_token_indices[:num_local_tokens] = local_token_indices.contiguous()
        padded_local_outputs = torch.zeros(
            (max_local_tokens, hidden_size),
            device=device,
            dtype=dtype,
        )
        padded_local_outputs[:num_local_tokens] = local_final_hidden_states.contiguous()

        gathered_token_indices = [
            torch.empty((cnt,), device=device, dtype=torch.int32) for cnt in gathered_counts
        ]
        gathered_outputs = [
            torch.empty((cnt, hidden_size), device=device, dtype=dtype) for cnt in gathered_counts
        ]

        dist.all_gather(gathered_token_indices, padded_local_token_indices)
        dist.all_gather(gathered_outputs, padded_local_outputs)

        final_hidden_states = torch.empty(
            (num_tokens, hidden_size),
            device=device,
            dtype=dtype,
        )

        for rank in range(self.ep_size):
            cnt = gathered_counts[rank]
            if cnt == 0:
                continue
            idx = gathered_token_indices[rank][:cnt].long()
            out = gathered_outputs[rank][:cnt]
            final_hidden_states[idx] = out

        return final_hidden_states

    def _forward_a2a(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, original_shape[-1])

        num_tokens = flat_hidden_states.shape[0]

        if num_tokens == 0:
            return flat_hidden_states.reshape(original_shape), None

        local_hidden_states, _local_token_indices, _ = self.shard_tokens(flat_hidden_states)
        router_logits_local = self.gate(local_hidden_states)  # [T_local, num_experts]
        topk_output = self.router(router_logits_local)
        topk_weights = topk_output.weights
        topk_ids_global = topk_output.ids

        dispatch_ctx = self._a2a_dispatch_tokens(
            flat_hidden_states=flat_hidden_states,
            topk_ids_global=topk_ids_global,
            topk_weights=topk_weights,
        )

        total_recv_slots = int(dispatch_ctx["total_recv_slots"])
        recv_hidden_states = dispatch_ctx["recv_hidden_states"]
        recv_local_expert = dispatch_ctx["recv_local_expert"]
        if total_recv_slots > 0:
            recv_topk_ids_local = recv_local_expert[:, None].contiguous()  # [R, 1]
            recv_topk_weights_local = torch.ones(
                (total_recv_slots, 1),
                device=recv_hidden_states.device,
                dtype=topk_weights.dtype,
            )
            recv_slot_outputs = self.expert_gemm(
                impl="triton",
                hidden_states=recv_hidden_states,
                w13=self.w13,
                w2=self.w2,
                topk_ids=recv_topk_ids_local,
                topk_weights=recv_topk_weights_local,
                local_expert_start=0,
                hidden_act=self.hidden_act,
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
