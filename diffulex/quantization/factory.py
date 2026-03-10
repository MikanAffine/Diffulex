"""
Quantization strategy factory.
"""

from typing import Optional

from diffulex.quantization.context import QuantizationContext
from diffulex.quantization.config import QuantizationConfig
from diffulex.quantization.registry import (
    create_kv_cache_strategy as _create_kv_cache_strategy,
    create_linear_strategy as _create_linear_strategy,
)
from diffulex.quantization.strategy import KVCacheQuantizationStrategy

from diffulex.quantization import strategies as _builtin_strategies  # noqa: F401


class QuantizationStrategyFactory:
    """Quantization strategy factory."""

    @staticmethod
    def create_kv_cache_strategy(
        dtype: Optional[str] = None,
    ) -> KVCacheQuantizationStrategy:
        return _create_kv_cache_strategy(dtype or "bf16")

    @staticmethod
    def create_from_config(config) -> QuantizationContext:
        ctx = QuantizationContext.current()
        quant_cfg = QuantizationConfig.from_diffulex_config(config)

        strategy = QuantizationStrategyFactory.create_kv_cache_strategy(quant_cfg.kv_cache.dtype)
        strategy.configure(diffulex_config=config)
        ctx.set_strategy("kv_cache", strategy)

        linear_attn = _create_linear_strategy(
            weight_dtype=quant_cfg.weights.linear_attn_dtype,
            act_dtype=quant_cfg.activations.linear_attn_dtype,
        )
        linear_attn.configure(diffulex_config=config)
        ctx.set_linear_strategy("attn", linear_attn)

        linear_mlp = _create_linear_strategy(
            weight_dtype=quant_cfg.weights.linear_mlp_dtype,
            act_dtype=quant_cfg.activations.linear_mlp_dtype,
        )
        linear_mlp.configure(diffulex_config=config)
        ctx.set_linear_strategy("mlp", linear_mlp)

        return ctx
