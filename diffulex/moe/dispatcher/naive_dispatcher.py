from __future__ import annotations

import torch
import torch.distributed as dist

from diffulex.moe.dispatcher.base_dispatcher import DispatcherOutput, TokenDispatcher
from diffulex.moe.metadata import DispatchMetadata


class NaiveA2ADispatcher(TokenDispatcher):
    """
    Naive A2A token dispatcher. Prototype of DeepEP-style dispatcher.
    """
    def _send_counts_from_dst_rank(self, dst_rank: torch.Tensor, *, device: torch.device) -> torch.Tensor:
        if dst_rank.numel() == 0:
            return torch.zeros((self.ep_size,), device=device, dtype=torch.int32)

        dst_rank_cpu = dst_rank.to(torch.int64).cpu()
        min_rank = int(dst_rank_cpu.min().item())
        max_rank = int(dst_rank_cpu.max().item())
        if min_rank < 0 or max_rank >= self.ep_size:
            raise ValueError(
                "MoE dispatcher produced an invalid destination rank: "
                f"min={min_rank}, max={max_rank}, ep_size={self.ep_size}, "
                f"num_local_experts={self.num_local_experts}."
            )

        return torch.bincount(dst_rank_cpu, minlength=self.ep_size).to(
            device=device,
            dtype=torch.int32,
        )

    @staticmethod
    def pack_dispatch_metadata(local_expert_ids: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        return (token_indices.to(torch.int64) << 32) | local_expert_ids.to(torch.int64)

    @staticmethod
    def unpack_dispatch_metadata(metadata: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_indices = torch.bitwise_right_shift(metadata, 32).to(torch.int32)
        local_expert_ids = torch.bitwise_and(metadata, 0xFFFFFFFF).to(torch.int32)
        return local_expert_ids, token_indices

    @staticmethod
    def pack_dispatch_payload(
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
    def unpack_dispatch_payload(
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
    def pack_return_payload(
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
    def unpack_return_payload(
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

    def __init__(
        self,
        *,
        ep_group,
        ep_size: int,
        num_local_experts: int,
        top_k: int,
        **_unused,
    ) -> None:
        self.ep_group = ep_group
        self.ep_size = ep_size
        self.num_local_experts = num_local_experts
        self.top_k = top_k

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor | None,
        topk_weights: torch.Tensor | None,
        *,
        active_dispatch: bool = True,
    ) -> DispatcherOutput:
        device = hidden_states.device
        dtype = hidden_states.dtype
        num_tokens, hidden_size = hidden_states.shape
        weight_dtype = dtype if topk_weights is None else topk_weights.dtype

        if active_dispatch:
            assert topk_ids is not None and topk_weights is not None
            if topk_weights.dtype != dtype:
                topk_weights = topk_weights.to(dtype)
            weight_dtype = dtype
            if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
                raise ValueError(
                    "MoE dispatcher expects topk_ids/topk_weights with matching [num_tokens, top_k] shape, "
                    f"got topk_ids={tuple(topk_ids.shape)}, topk_weights={tuple(topk_weights.shape)}."
                )
            actual_top_k = int(topk_ids.shape[1])
            if int(topk_ids.shape[0]) != num_tokens:
                raise ValueError(
                    "MoE dispatcher topk metadata does not match hidden_states: "
                    f"num_tokens={num_tokens}, topk_ids.shape={tuple(topk_ids.shape)}."
                )
            if topk_ids.numel() > 0:
                topk_ids_cpu = topk_ids.to(torch.int64).cpu()
                min_expert = int(topk_ids_cpu.min().item())
                max_expert = int(topk_ids_cpu.max().item())
                max_valid_expert = self.ep_size * self.num_local_experts
                if min_expert < 0 or max_expert >= max_valid_expert:
                    raise ValueError(
                        "MoE router produced an invalid expert id: "
                        f"min={min_expert}, max={max_expert}, valid=[0,{max_valid_expert}), "
                        f"ep_size={self.ep_size}, num_local_experts={self.num_local_experts}."
                    )
            local_token_indices = torch.arange(num_tokens, device=device, dtype=torch.int32)
            slot_hidden_states = hidden_states.repeat_interleave(actual_top_k, dim=0).contiguous()
            slot_token_indices = (
                local_token_indices.unsqueeze(1).expand(num_tokens, actual_top_k).reshape(-1).contiguous()
            )
            slot_expert_ids_global = topk_ids.reshape(-1).contiguous()
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
            send_metadata = self.pack_dispatch_metadata(send_local_expert, send_token_indices).contiguous()
            send_counts = self._send_counts_from_dst_rank(dst_rank_sorted, device=device)
        else:
            send_hidden_states = torch.empty((0, hidden_size), device=device, dtype=dtype)
            send_metadata = torch.empty((0,), device=device, dtype=torch.int64)
            send_weights = torch.empty((0,), device=device, dtype=weight_dtype)
            send_counts = torch.zeros((self.ep_size,), device=device, dtype=torch.int32)

        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        send_splits = send_counts.cpu().tolist()
        recv_splits = recv_counts.cpu().tolist()
        total_recv_slots = int(recv_counts.sum().item())
        recv_hidden_states = torch.empty((total_recv_slots, hidden_size), device=device, dtype=dtype)
        recv_metadata = torch.empty((total_recv_slots,), device=device, dtype=torch.int64)
        recv_weights = torch.empty((total_recv_slots,), device=device, dtype=weight_dtype)
        dist.all_to_all_single(
            recv_hidden_states,
            send_hidden_states,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.ep_group,
        )
        dist.all_to_all_single(
            recv_metadata,
            send_metadata,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.ep_group,
        )
        dist.all_to_all_single(
            recv_weights,
            send_weights,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.ep_group,
        )
        recv_local_expert, recv_token_indices = self.unpack_dispatch_metadata(recv_metadata)
        metadata = DispatchMetadata(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
            send_splits=send_splits,
            recv_splits=recv_splits,
            recv_hidden_states=recv_hidden_states,
            recv_local_expert=recv_local_expert,
            recv_token_indices=recv_token_indices,
            recv_weights=recv_weights,
            total_recv_slots=total_recv_slots,
            active_dispatch=active_dispatch,
        )
        return DispatcherOutput(
            metadata=metadata,
            recv_hidden_states=recv_hidden_states,
            recv_local_expert_ids=recv_local_expert,
            recv_weights=recv_weights,
        )

    def combine(
        self,
        slot_outputs: torch.Tensor,
        metadata: DispatchMetadata,
    ) -> torch.Tensor:
        if slot_outputs.dtype != metadata.dtype:
            slot_outputs = slot_outputs.to(metadata.dtype)
        total_returned_slots = int(sum(metadata.send_splits))
        send_back_outputs = slot_outputs.contiguous()
        send_back_token_indices = metadata.recv_token_indices.to(torch.int32).contiguous()
        returned_outputs = torch.empty(
            (total_returned_slots, metadata.hidden_size),
            device=metadata.device,
            dtype=metadata.dtype,
        )
        returned_token_indices = torch.empty(
            (total_returned_slots,),
            device=metadata.device,
            dtype=torch.int32,
        )
        dist.all_to_all_single(
            returned_outputs,
            send_back_outputs,
            output_split_sizes=metadata.send_splits,
            input_split_sizes=metadata.recv_splits,
            group=self.ep_group,
        )
        dist.all_to_all_single(
            returned_token_indices,
            send_back_token_indices,
            output_split_sizes=metadata.send_splits,
            input_split_sizes=metadata.recv_splits,
            group=self.ep_group,
        )
        if returned_token_indices.numel() > 0:
            returned_token_indices_cpu = returned_token_indices.to(torch.int64).cpu()
            min_token_idx = int(returned_token_indices_cpu.min().item())
            max_token_idx = int(returned_token_indices_cpu.max().item())
            if min_token_idx < 0 or max_token_idx >= metadata.num_tokens:
                raise ValueError(
                    "MoE combine received invalid token indices: "
                    f"min={min_token_idx}, max={max_token_idx}, num_tokens={metadata.num_tokens}, "
                    f"send_splits={metadata.send_splits}, recv_splits={metadata.recv_splits}, "
                    f"returned_slots={returned_token_indices.numel()}."
                )
        final_hidden_states = torch.zeros(
            (metadata.num_tokens, metadata.hidden_size),
            device=metadata.device,
            dtype=torch.float32,
        )
        if total_returned_slots > 0:
            final_hidden_states.index_add_(0, returned_token_indices.long(), returned_outputs.float())
        return final_hidden_states.to(metadata.dtype).contiguous()

__all__ = ["NaiveA2ADispatcher"]
