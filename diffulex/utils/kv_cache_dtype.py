"""
KV Cache dtype utilities.

Canonical implementation lives in diffulex.quantization.kv_cache_dtype.
This module re-exports for backward compatibility.
"""

from diffulex.quantization.kv_cache_dtype import (
    kv_cacheDType,
    kv_cacheDTypeSpec,
    parse_kv_cache_dtype,
    ensure_scale_tensor,
    view_fp8_cache,
    _normalize_kv_cache_dtype,
    _get_fp8_e4m3_dtype,
    _get_fp8_e5m2_dtype,
)

__all__ = [
    "kv_cacheDType",
    "kv_cacheDTypeSpec",
    "parse_kv_cache_dtype",
    "ensure_scale_tensor",
    "view_fp8_cache",
]
