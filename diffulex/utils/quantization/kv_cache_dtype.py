"""Re-export from diffulex.quantization.kv_cache_dtype."""

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
