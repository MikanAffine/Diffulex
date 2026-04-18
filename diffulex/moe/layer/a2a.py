from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from diffulex.moe.layer.ep_impl import EPFusedMoE


@dataclass(frozen=True)
class DispatchState:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    token_ids: torch.Tensor
    num_tokens: int
    local_num_tokens: int
    topk_ids_are_local: bool
    local_expert_start: int


class A2ABackend(ABC):
    name = "base"

    def __init__(self, moe: "EPFusedMoE") -> None:
        self.moe = moe
        self.ep_rank = moe.ep_rank
        self.ep_size = moe.ep_size
        self.hidden_size = moe.hidden_size
        self.num_local_experts = moe.num_local_experts
        self.local_expert_start = moe.local_expert_start

    @staticmethod
    def _num_tokens_for_rank(num_tokens: int, rank: int, ep_size: int) -> int:
        remaining = num_tokens - rank
        if remaining <= 0:
            return 0
        return (remaining + ep_size - 1) // ep_size

    def _source_block_sizes(self, num_tokens: int, top_k: int) -> list[int]:
        return [
            self._num_tokens_for_rank(num_tokens, rank, self.ep_size) * top_k
            for rank in range(self.ep_size)
        ]

    def _own_block_size(self, num_tokens: int, top_k: int) -> int:
        return self._num_tokens_for_rank(num_tokens, self.ep_rank, self.ep_size) * top_k

    @staticmethod
    def _all_to_all_tensor(
        send_tensor: torch.Tensor,
        *,
        send_splits: list[int],
        recv_splits: list[int],
    ) -> tuple[torch.Tensor, object | None]:
        recv_shape = (sum(recv_splits), *send_tensor.shape[1:])
        recv_tensor = torch.empty(
            recv_shape,
            device=send_tensor.device,
            dtype=send_tensor.dtype,
        )
        work = dist.all_to_all_single(
            recv_tensor,
            send_tensor.contiguous(),
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            async_op=True,
        )
        return recv_tensor, work

    def _gather_local_outputs(
        self,
        local_final_hidden_states: torch.Tensor,
        *,
        num_tokens: int,
    ) -> torch.Tensor:
        if self.ep_size == 1:
            return local_final_hidden_states

        num_local_tokens, hidden_size = local_final_hidden_states.shape
        max_local_tokens = self._num_tokens_for_rank(num_tokens, 0, self.ep_size)
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
        work = dist.all_gather(gathered_outputs, padded_local_outputs, async_op=True)
        work.wait()

        final_hidden_states = torch.empty(
            (num_tokens, hidden_size),
            device=local_final_hidden_states.device,
            dtype=local_final_hidden_states.dtype,
        )
        for rank in range(self.ep_size):
            local_count = self._num_tokens_for_rank(num_tokens, rank, self.ep_size)
            if local_count == 0:
                continue
            final_hidden_states[rank::self.ep_size] = gathered_outputs[rank][:local_count]
        return final_hidden_states

    @abstractmethod
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> DispatchState:
        raise NotImplementedError

    @abstractmethod
    def combine(
        self,
        local_outputs: torch.Tensor,
        dispatch_state: DispatchState,
    ) -> torch.Tensor:
        raise NotImplementedError


class NoneA2ABackend(A2ABackend):
    name = "none"

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> DispatchState:
        num_tokens = hidden_states.shape[0]
        local_num_tokens = num_tokens
        token_ids = torch.arange(
            num_tokens,
            device=hidden_states.device,
            dtype=torch.int64,
        ).repeat_interleave(topk_ids.shape[1])
        return DispatchState(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            token_ids=token_ids,
            num_tokens=num_tokens,
            local_num_tokens=local_num_tokens,
            topk_ids_are_local=False,
            local_expert_start=self.local_expert_start,
        )

    def combine(
        self,
        local_outputs: torch.Tensor,
        dispatch_state: DispatchState,
    ) -> torch.Tensor:
        if self.ep_size > 1:
            dist.all_reduce(local_outputs)
        return local_outputs


class A2ATokenDispatchBackend(A2ABackend):
    name = "a2a"

    def _pack_dispatch_payload(
        self,
        *,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        local_expert_ids: torch.Tensor,
        weights: torch.Tensor,
        dest_ranks: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        payload_dim = self.hidden_size + 3
        payload = torch.zeros(
            (self.ep_size, block_size, payload_dim),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        payload[:, :, self.hidden_size : self.hidden_size + 2] = -1.0

        if block_size == 0:
            return payload.reshape(0, payload_dim)

        sort_order = torch.argsort(dest_ranks)
        sorted_dest_ranks = dest_ranks.index_select(0, sort_order)
        sorted_hidden_states = hidden_states.index_select(0, sort_order).to(torch.float32)
        sorted_token_ids = token_ids.index_select(0, sort_order).to(torch.float32)
        sorted_local_expert_ids = local_expert_ids.index_select(0, sort_order).to(torch.float32)
        sorted_weights = weights.index_select(0, sort_order).to(torch.float32)

        dest_counts = torch.bincount(sorted_dest_ranks, minlength=self.ep_size)
        dest_offsets = torch.cumsum(dest_counts, dim=0) - dest_counts
        dest_counts_list = dest_counts.tolist()
        dest_offsets_list = dest_offsets.tolist()

        for dest_rank in range(self.ep_size):
            count = int(dest_counts_list[dest_rank])
            if count == 0:
                continue
            start = int(dest_offsets_list[dest_rank])
            end = start + count
            segment = payload[dest_rank]
            segment[:count, : self.hidden_size] = sorted_hidden_states[start:end]
            segment[:count, self.hidden_size] = sorted_token_ids[start:end]
            segment[:count, self.hidden_size + 1] = sorted_local_expert_ids[start:end]
            segment[:count, self.hidden_size + 2] = sorted_weights[start:end]

        return payload.reshape(self.ep_size * block_size, payload_dim)

    def _unpack_dispatch_payload(
        self,
        recv_tensor: torch.Tensor,
        *,
        source_block_sizes: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        payload_dim = self.hidden_size + 3
        hidden_chunks: list[torch.Tensor] = []
        token_id_chunks: list[torch.Tensor] = []
        expert_id_chunks: list[torch.Tensor] = []
        weight_chunks: list[torch.Tensor] = []

        offset = 0
        for block_size in source_block_sizes:
            block = recv_tensor[offset : offset + block_size]
            offset += block_size
            if block_size == 0:
                continue

            valid_rows = block[block[:, self.hidden_size] >= 0]
            if valid_rows.numel() == 0:
                continue

            hidden_chunks.append(valid_rows[:, : self.hidden_size])
            token_id_chunks.append(valid_rows[:, self.hidden_size].to(torch.int64))
            expert_id_chunks.append(valid_rows[:, self.hidden_size + 1].to(torch.int32))
            weight_chunks.append(valid_rows[:, self.hidden_size + 2])

        if not hidden_chunks:
            device = recv_tensor.device
            empty_hidden = torch.empty((0, self.hidden_size), device=device, dtype=torch.float32)
            empty_token_ids = torch.empty((0,), device=device, dtype=torch.int64)
            empty_expert_ids = torch.empty((0,), device=device, dtype=torch.int32)
            empty_weights = torch.empty((0,), device=device, dtype=torch.float32)
            return empty_hidden, empty_token_ids, empty_expert_ids, empty_weights

        hidden_states = torch.cat(hidden_chunks, dim=0)
        token_ids = torch.cat(token_id_chunks, dim=0)
        local_expert_ids = torch.cat(expert_id_chunks, dim=0)
        weights = torch.cat(weight_chunks, dim=0)
        return hidden_states, token_ids, local_expert_ids, weights

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> DispatchState:
        num_tokens = hidden_states.shape[0]
        top_k = topk_ids.shape[1]
        device = hidden_states.device

        local_token_ids = torch.arange(
            self.ep_rank,
            num_tokens,
            self.ep_size,
            device=device,
            dtype=torch.int64,
        )
        local_hidden_states = hidden_states.index_select(0, local_token_ids)
        local_topk_ids = topk_ids.index_select(0, local_token_ids)
        local_topk_weights = topk_weights.index_select(0, local_token_ids)

        local_num_tokens = local_hidden_states.shape[0]
        local_block_size = local_num_tokens * top_k
        source_block_sizes = self._source_block_sizes(num_tokens, top_k)
        own_block_size = source_block_sizes[self.ep_rank]

        flat_token_ids = local_token_ids.repeat_interleave(top_k)
        flat_hidden_states = local_hidden_states.repeat_interleave(top_k, dim=0)
        flat_global_expert_ids = local_topk_ids.reshape(-1).to(torch.int64)
        flat_weights = local_topk_weights.reshape(-1)

        dest_ranks = torch.div(
            flat_global_expert_ids,
            self.num_local_experts,
            rounding_mode="floor",
        ).to(torch.int64)
        local_expert_ids = (
            flat_global_expert_ids - dest_ranks * self.num_local_experts
        ).to(torch.int32)

        send_tensor = self._pack_dispatch_payload(
            hidden_states=flat_hidden_states,
            token_ids=flat_token_ids,
            local_expert_ids=local_expert_ids,
            weights=flat_weights,
            dest_ranks=dest_ranks,
            block_size=local_block_size,
        )

        recv_tensor, work = self._all_to_all_tensor(
            send_tensor,
            send_splits=[local_block_size] * self.ep_size,
            recv_splits=source_block_sizes,
        )
        work.wait()

        hidden_states_recv, token_ids_recv, local_expert_ids_recv, weights_recv = self._unpack_dispatch_payload(
            recv_tensor,
            source_block_sizes=source_block_sizes,
        )

        return DispatchState(
            hidden_states=hidden_states_recv.to(self.moe.w13.dtype),
            topk_ids=local_expert_ids_recv.unsqueeze(-1),
            topk_weights=weights_recv.unsqueeze(-1).to(self.moe.w13.dtype),
            token_ids=token_ids_recv,
            num_tokens=num_tokens,
            local_num_tokens=local_num_tokens,
            topk_ids_are_local=True,
            local_expert_start=0,
        )

    def _pack_return_payload(
        self,
        *,
        local_outputs: torch.Tensor,
        token_ids: torch.Tensor,
        source_block_sizes: list[int],
    ) -> torch.Tensor:
        payload_dim = self.hidden_size + 1
        payload = torch.zeros(
            (sum(source_block_sizes), payload_dim),
            device=local_outputs.device,
            dtype=torch.float32,
        )
        payload[:, self.hidden_size] = -1.0

        if local_outputs.numel() == 0:
            return payload

        source_ids = torch.remainder(token_ids, self.ep_size).to(torch.int64)
        sort_order = torch.argsort(source_ids)
        sorted_source_ids = source_ids.index_select(0, sort_order)
        sorted_local_outputs = local_outputs.index_select(0, sort_order).to(torch.float32)
        sorted_token_ids = token_ids.index_select(0, sort_order).to(torch.float32)

        source_counts = torch.bincount(sorted_source_ids, minlength=self.ep_size)
        source_offsets = torch.cumsum(source_counts, dim=0) - source_counts
        source_block_sizes_tensor = torch.tensor(
            source_block_sizes,
            device=local_outputs.device,
            dtype=torch.int64,
        )
        source_starts = torch.cumsum(source_block_sizes_tensor, dim=0) - source_block_sizes_tensor
        source_counts_list = source_counts.tolist()
        source_offsets_list = source_offsets.tolist()
        source_starts_list = source_starts.tolist()

        for source_rank in range(self.ep_size):
            count = int(source_counts_list[source_rank])
            if count == 0:
                continue
            start = int(source_offsets_list[source_rank])
            end = start + count
            block_start = int(source_starts_list[source_rank])
            segment = payload[block_start : block_start + source_block_sizes[source_rank]]
            segment[:count, : self.hidden_size] = sorted_local_outputs[start:end]
            segment[:count, self.hidden_size] = sorted_token_ids[start:end]

        return payload

    def _unpack_return_payload(
        self,
        recv_tensor: torch.Tensor,
        *,
        own_block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        payload_dim = self.hidden_size + 1
        if own_block_size == 0:
            device = recv_tensor.device
            empty_hidden = torch.empty((0, self.hidden_size), device=device, dtype=torch.float32)
            empty_token_ids = torch.empty((0,), device=device, dtype=torch.int64)
            return empty_hidden, empty_token_ids

        recv_tensor = recv_tensor.view(self.ep_size, own_block_size, payload_dim)
        hidden_chunks: list[torch.Tensor] = []
        token_id_chunks: list[torch.Tensor] = []
        for source_rank in range(self.ep_size):
            block = recv_tensor[source_rank]
            valid_rows = block[block[:, self.hidden_size] >= 0]
            if valid_rows.numel() == 0:
                continue
            hidden_chunks.append(valid_rows[:, : self.hidden_size])
            token_id_chunks.append(valid_rows[:, self.hidden_size].to(torch.int64))

        if not hidden_chunks:
            device = recv_tensor.device
            empty_hidden = torch.empty((0, self.hidden_size), device=device, dtype=torch.float32)
            empty_token_ids = torch.empty((0,), device=device, dtype=torch.int64)
            return empty_hidden, empty_token_ids

        return torch.cat(hidden_chunks, dim=0), torch.cat(token_id_chunks, dim=0)

    def combine(
        self,
        local_outputs: torch.Tensor,
        dispatch_state: DispatchState,
    ) -> torch.Tensor:
        source_block_sizes = self._source_block_sizes(dispatch_state.num_tokens, self.moe.topk)
        own_block_size = source_block_sizes[self.ep_rank]

        send_tensor = self._pack_return_payload(
            local_outputs=local_outputs,
            token_ids=dispatch_state.token_ids,
            source_block_sizes=source_block_sizes,
        )

        recv_tensor, work = self._all_to_all_tensor(
            send_tensor,
            send_splits=source_block_sizes,
            recv_splits=[own_block_size] * self.ep_size,
        )
        work.wait()

        returned_hidden_states, returned_token_ids = self._unpack_return_payload(
            recv_tensor,
            own_block_size=own_block_size,
        )

        local_num_tokens = dispatch_state.local_num_tokens
        local_final_hidden_states = torch.zeros(
            (local_num_tokens, self.hidden_size),
            device=local_outputs.device,
            dtype=returned_hidden_states.dtype,
        )
        if returned_hidden_states.numel() > 0:
            local_token_indices = torch.div(
                returned_token_ids.to(torch.int64),
                self.ep_size,
                rounding_mode="floor",
            )
            local_final_hidden_states.index_add_(
                0,
                local_token_indices,
                returned_hidden_states,
            )

        return self._gather_local_outputs(
            local_final_hidden_states,
            num_tokens=dispatch_state.num_tokens,
        )


class DeepEPA2ABackend(A2ATokenDispatchBackend):
    name = "deepep"


_BACKEND_REGISTRY: dict[str, type[A2ABackend]] = {
    NoneA2ABackend.name: NoneA2ABackend,
    A2ATokenDispatchBackend.name: A2ATokenDispatchBackend,
    DeepEPA2ABackend.name: DeepEPA2ABackend,
    "replicated": NoneA2ABackend,
}


def build_a2a_backend(kind: str | None, moe: "EPFusedMoE") -> A2ABackend:
    key = "none" if kind is None else str(kind).lower()
    backend_cls = _BACKEND_REGISTRY.get(key)
    if backend_cls is None:
        raise NotImplementedError(f"Unknown A2A backend: {kind!r}.")
    return backend_cls(moe)


__all__ = [
    "A2ABackend",
    "A2ATokenDispatchBackend",
    "DeepEPA2ABackend",
    "DispatchState",
    "NoneA2ABackend",
    "build_a2a_backend",
]
