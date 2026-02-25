"""
Backward compatibility: quantization now lives under diffulex.quantization.
Re-export from the canonical location.
"""

from diffulex.quantization import (
    QuantizationContext,
    get_quantization_context,
    set_kv_cache_strategy,
    get_kv_cache_strategy,
    QuantizationStrategyFactory,
    QuantizationConfig,
    kv_cacheQuantConfig,
    WeightQuantConfig,
    ActivationQuantConfig,
    create_kv_cache_strategy,
    registered_kv_cache_dtypes,
    QuantizationStrategy,
    KVCacheQuantizationStrategy,
    WeightQuantizationStrategy,
    kv_cacheDType,
    kv_cacheDTypeSpec,
    parse_kv_cache_dtype,
    ensure_scale_tensor,
    view_fp8_cache,
)

__all__ = [
    "QuantizationContext",
    "get_quantization_context",
    "set_kv_cache_strategy",
    "get_kv_cache_strategy",
    "QuantizationStrategyFactory",
    "QuantizationConfig",
    "kv_cacheQuantConfig",
    "WeightQuantConfig",
    "ActivationQuantConfig",
    "create_kv_cache_strategy",
    "registered_kv_cache_dtypes",
    "QuantizationStrategy",
    "kv_cacheQuantizationStrategy",
    "WeightQuantizationStrategy",
    "kv_cacheDType",
    "kv_cacheDTypeSpec",
    "parse_kv_cache_dtype",
    "ensure_scale_tensor",
    "view_fp8_cache",
]
