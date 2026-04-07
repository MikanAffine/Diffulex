from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LoadContext:
    config: Any
    full_name: str


@dataclass(frozen=True)
class ResolvedWeight:
    param: nn.Parameter | None = None
    buffer: torch.Tensor | None = None
    shard_id: int | str | None = None
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None
    loader: Callable[[torch.Tensor], None] | None = None
    skip: bool = False


__all__ = [
    "LoadContext",
    "ResolvedWeight",
]
