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
    ):
        self.use_multi_block = True
        self.valid_slices = valid_slices
        self.buffer_size = buffer_size
        
        if self.page_size % self.diffusion_block_size != 0:
            raise ValueError(f"page_size {self.page_size} must be divisible by diffusion_block_size {self.diffusion_block_size}")
        
        if self.attn_type != "block_attention":
            raise ValueError(f"attn_type must be block_attention, got {self.attn_type}")