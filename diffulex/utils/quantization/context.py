"""Re-export from diffulex.quantization.context."""

from diffulex.quantization.context import (
    QuantizationContext,
    get_quantization_context,
    set_kv_cache_strategy,
    get_kv_cache_strategy,
    set_weight_strategy,
    get_weight_strategy,
    set_linear_strategy,
    get_linear_strategy,
    clear_act_quant_cache,
    get_cached_act_quant,
    set_cached_act_quant,
)

__all__ = [
    "QuantizationContext",
    "get_quantization_context",
    "set_kv_cache_strategy",
    "get_kv_cache_strategy",
    "set_weight_strategy",
    "get_weight_strategy",
    "set_linear_strategy",
    "get_linear_strategy",
    "clear_act_quant_cache",
    "get_cached_act_quant",
    "set_cached_act_quant",
]
