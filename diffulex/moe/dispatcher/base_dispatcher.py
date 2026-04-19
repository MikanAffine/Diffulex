from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from diffulex.moe.metadata import DispatchMetadata


@dataclass(frozen=True)
class DispatcherOutput:
    metadata: DispatchMetadata
    recv_hidden_states: torch.Tensor
    recv_local_expert_ids: torch.Tensor
    recv_weights: torch.Tensor


class TokenDispatcher(ABC):
    def set_forward_phase(self, phase: str) -> None:
        del phase

    @abstractmethod
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor | None,
        topk_weights: torch.Tensor | None,
        *,
        active_dispatch: bool = True,
    ) -> DispatcherOutput:
        raise NotImplementedError

    @abstractmethod
    def combine(
        self,
        slot_outputs: torch.Tensor,
        metadata: DispatchMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError


def build_token_dispatcher(
    backend: str,
    **kwargs,
) -> TokenDispatcher:
    if backend == "standard":
        raise NotImplementedError(
            "moe_dispatcher_backend='standard' does not build an EP token dispatcher. "
            "Use TPFusedMoE with expert_parallel_size=1, or choose 'naive'/'deepep' for EP A2A."
        )
    if backend == "naive":
        from diffulex.moe.dispatcher.naive_dispatcher import NaiveA2ADispatcher

        return NaiveA2ADispatcher(**kwargs)
    if backend == "deepep":
        from diffulex.moe.dispatcher.deepep_dispatcher import DeepEPDispatcher

        return DeepEPDispatcher(**kwargs)
    raise NotImplementedError(f"Unsupported MoE dispatcher backend: {backend}")


__all__ = [
    "build_token_dispatcher",
    "DispatcherOutput",
    "TokenDispatcher",
]
