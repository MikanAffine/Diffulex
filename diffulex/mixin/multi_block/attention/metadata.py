from __future__ import annotations

import torch

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffulex.attention.metadata import AttnMetaDataBase


@dataclass
class MultiBlockAttnMetaDataMixin:
    def init_multi_block(
        self: AttnMetaDataBase,
        valid_slices: torch.Tensor | None = None,
        buffer_size: int = 1,
        is_prefix_full: bool = False,
        status_table: torch.Tensor | None = None,
        prefix_lens: torch.Tensor | None = None,
        padded_prefix_lens: torch.Tensor | None = None,
    ):
        self.use_multi_block = True
        self.valid_slices = valid_slices
        self.buffer_size = buffer_size
        self.is_block_causal = True
        self.is_prefix_full = is_prefix_full
        self.status_table = status_table
        self.prefix_lens = prefix_lens
        self.padded_prefix_lens = padded_prefix_lens

        if self.page_size % self.block_size != 0:
            raise ValueError(f"page_size {self.page_size} must be divisible by block_size {self.block_size}")
