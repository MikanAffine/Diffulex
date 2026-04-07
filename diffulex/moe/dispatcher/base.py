from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from diffulex.moe.dispatcher.datatype import CombineInput, DispatchOutput
from diffulex.moe.topk import TopKOutput


class MoEDispatcher(nn.Module, ABC):
    """Dispatches tokens to experts / Combines tokens from experts efficiently."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    @abstractmethod
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        *,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> DispatchOutput:
        raise NotImplementedError

    @abstractmethod
    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        raise NotImplementedError


TokenDispatcher = MoEDispatcher
