"""
KV Cache dtype utilities.

Canonical implementation lives in diffulex.quantization.kv_cache_dtype.
This module re-exports for backward compatibility.
"""

from diffulex.quantization.kv_cache_dtype import (
    KVCacheDType,
    KVCacheDTypeSpec,
    parse_kv_cache_dtype,
    ensure_scale_tensor,
    view_fp8_cache,
)

__all__ = [
    "KVCacheDType",
    "KVCacheDTypeSpec",
    "parse_kv_cache_dtype",
    "ensure_scale_tensor",
    "view_fp8_cache",
]
