from __future__ import annotations

import torch

from diffulex.attention.metadata import AttnMetaDataBase
from diffulex.utils.quantization.context import get_kv_cache_strategy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffulex.attention.attn_impl import Attention


class AttentionQuantizationMixin:
    def use_quantization(self: Attention) -> bool:
        return self.k_scale is not None and self.v_scale is not None

    def maybe_update_scales(self: Attention, k: torch.Tensor, v: torch.Tensor, attn_metadata: AttnMetaDataBase) -> None:
        # Update scales if quantization strategy requires them
        strategy = get_kv_cache_strategy()
        if strategy is not None:
            self.k_scale, self.v_scale = strategy.update_scales(
                k, v, self.k_scale, self.v_scale, self.num_kv_heads, k.device
            )

    def maybe_set_attn_metadata_scales(self: Attention, attn_metadata: AttnMetaDataBase) -> None:
        strategy = get_kv_cache_strategy()
        # Pass scale to metadata if required by strategy
        if strategy is not None:
            strategy.maybe_set_attn_metadata_scales(attn_metadata, k_scale=self.k_scale, v_scale=self.v_scale)
