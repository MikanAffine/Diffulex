"""
Quantization module for diffulex.

All quantization logic lives under diffulex.quantization. This package provides:
- KV Cache quantization
- Linear (weight + activation) quantization
- Strategy pattern with context management and mixins for engine/layer integration.
"""

from diffulex.quantization.context import (
    QuantizationContext,
    get_quantization_context,
    set_kv_cache_strategy,
    get_kv_cache_strategy,
)
from diffulex.quantization.factory import QuantizationStrategyFactory
from diffulex.quantization.config import (
    QuantizationConfig,
    KVCacheQuantConfig,
    WeightQuantConfig,
    ActivationQuantConfig,
)
from diffulex.quantization.registry import (
    create_kv_cache_strategy,
    registered_kv_cache_dtypes,
)
from diffulex.quantization.strategy import (
    QuantizationStrategy,
    KVCacheQuantizationStrategy,
    WeightQuantizationStrategy,
)
from diffulex.quantization.kv_cache_dtype import (
    KVCacheDType,
    KVCacheDTypeSpec,
    parse_kv_cache_dtype,
    ensure_scale_tensor,
    view_fp8_cache,
)

__all__ = [
    # Context
    "QuantizationContext",
    "get_quantization_context",
    "set_kv_cache_strategy",
    "get_kv_cache_strategy",
    # Factory
    "QuantizationStrategyFactory",
    # Config
    "QuantizationConfig",
    "KVCacheQuantConfig",
    "WeightQuantConfig",
    "ActivationQuantConfig",
    # Registry
    "create_kv_cache_strategy",
    "registered_kv_cache_dtypes",
    # Strategy interfaces
    "QuantizationStrategy",
    "KVCacheQuantizationStrategy",
    "WeightQuantizationStrategy",
    # KV Cache dtype utilities
    "KVCacheDType",
    "KVCacheDTypeSpec",
    "parse_kv_cache_dtype",
    "ensure_scale_tensor",
    "view_fp8_cache",
]
