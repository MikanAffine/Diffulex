"""Re-export from diffulex.quantization.registry."""
from diffulex.quantization.registry import (
    create_kv_cache_strategy,
    registered_kv_cache_dtypes,
    register_kv_cache_strategy,
    create_linear_strategy,
    register_linear_strategy,
    registered_linear_dtypes,
)

__all__ = [
    "create_kv_cache_strategy",
    "registered_kv_cache_dtypes",
    "register_kv_cache_strategy",
    "create_linear_strategy",
    "register_linear_strategy",
    "registered_linear_dtypes",
]
