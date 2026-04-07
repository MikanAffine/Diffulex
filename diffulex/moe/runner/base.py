from __future__ import annotations

from collections.abc import Callable
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from diffulex.moe.dispatcher import CombineInput, DispatchOutput


class MoERunner(nn.Module, ABC):
    """Runs MoE MLP GEMMs efficiently."""

    def __init__(
        self,
        *,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        top_k: int | None = None,
        hidden_act: str = "silu",
        get_w13: Callable[[], torch.Tensor] | None = None,
        get_w2: Callable[[], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.hidden_act = hidden_act
        self._get_w13 = get_w13
        self._get_w2 = get_w2

    @property
    def w13(self) -> torch.Tensor:
        if self._get_w13 is None:
            raise RuntimeError("MoE runner w13 weights are not configured.")
        return self._get_w13()

    @property
    def w2(self) -> torch.Tensor:
        if self._get_w2 is None:
            raise RuntimeError("MoE runner w2 weights are not configured.")
        return self._get_w2()

    @abstractmethod
    def forward(self, dispatch_output: DispatchOutput) -> CombineInput:
        raise NotImplementedError
