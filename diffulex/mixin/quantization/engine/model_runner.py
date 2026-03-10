"""
Quantization mixin for model runner.

Provides initialization of quantization context from config and KV cache strategy
for allocation. The engine model_runner should inherit ModelRunnerQuantizationMixin
and call init_quantization_from_config(config) in __init__, then use
get_kv_cache_strategy_for_alloc() in allocate_kv_cache().
"""

import torch

from typing import Any

from diffulex.quantization.factory import QuantizationStrategyFactory
from diffulex.quantization.context import get_kv_cache_strategy
from diffulex.quantization.strategies import NoQuantizationStrategy
from diffulex.quantization.strategy import KVCacheQuantizationStrategy


class ModelRunnerQuantizationMixin:
    """Mixin for model runner quantization: context init and KV cache strategy."""

    def init_quantization_from_config(self, config: Any) -> None:
        """Initialize quantization context from Diffulex config. Call once in runner __init__."""
        QuantizationStrategyFactory.create_from_config(config)

    def get_kv_cache_strategy_for_alloc(self) -> KVCacheQuantizationStrategy:
        """Return KV cache quantization strategy for allocation; never None."""
        strategy = get_kv_cache_strategy()
        if strategy is None:
            return NoQuantizationStrategy()
        return strategy

    def init_quantization_scales(self):
        config = self.config
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads
        device = self.k_cache.device if config.kv_cache_layout == "distinct" else self.kv_cache.device
        strategy = get_kv_cache_strategy()

        attn_modules = [m for m in self.model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache")]

        k_scale_init, v_scale_init = strategy.init_scales(num_kv_heads, device)
        if k_scale_init is not None and v_scale_init is not None:
            # Allocate scale tensors: [num_layers, num_kv_heads]
            self.k_scale = torch.zeros(
                hf_config.num_hidden_layers,
                num_kv_heads,
                dtype=torch.float32,
                device=device,
            )
            self.v_scale = torch.zeros(
                hf_config.num_hidden_layers,
                num_kv_heads,
                dtype=torch.float32,
                device=device,
            )
            # Initialize with strategy's initial scale values
            self.k_scale[:] = k_scale_init[None, :]
            self.v_scale[:] = v_scale_init[None, :]

            # Bind scales to Attention modules
            for layer_id, module in enumerate(attn_modules):
                module.k_scale = self.k_scale[layer_id]
                module.v_scale = self.v_scale[layer_id]
