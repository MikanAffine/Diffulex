from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TopKOutput:
    weights: torch.Tensor | None
    ids: torch.Tensor | None
    router_logits: torch.Tensor
