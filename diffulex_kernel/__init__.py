"""Diffulex CUDA kernel package.

Keep this module lightweight: importing `diffulex_kernel` should not eagerly
import optional heavy deps (e.g. TileLang) unless the corresponding kernels are
actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffulex_kernel.python.dllm_flash_attn_kernels import (  # noqa: F401
        dllm_flash_attn_decode as dllm_flash_attn_decode,
        dllm_flash_attn_prefill as dllm_flash_attn_prefill,
    )
    from diffulex_kernel.python.kv_cache_kernels import (  # noqa: F401
        load_kv_cache as load_kv_cache,
        store_kv_cache_distinct_layout as store_kv_cache_distinct_layout,
        store_kv_cache_unified_layout as store_kv_cache_unified_layout,
    )


def __getattr__(name: str):
    if name == "dllm_flash_attn_decode":
        from diffulex_kernel.python.dllm_flash_attn_kernels import dllm_flash_attn_decode

        return dllm_flash_attn_decode
    if name == "dllm_flash_attn_prefill":
        from diffulex_kernel.python.dllm_flash_attn_kernels import dllm_flash_attn_prefill

        return dllm_flash_attn_prefill
    if name == "store_kv_cache_distinct_layout":
        from diffulex_kernel.python.kv_cache_kernels import store_kv_cache_distinct_layout

        return store_kv_cache_distinct_layout
    if name == "store_kv_cache_unified_layout":
        from diffulex_kernel.python.kv_cache_kernels import store_kv_cache_unified_layout

        return store_kv_cache_unified_layout
    if name == "load_kv_cache":
        from diffulex_kernel.python.kv_cache_kernels import load_kv_cache

        return load_kv_cache
    raise AttributeError(name)


__all__ = [
    "dllm_flash_attn_decode",
    "dllm_flash_attn_prefill",
    "store_kv_cache_distinct_layout",
    "store_kv_cache_unified_layout",
    "load_kv_cache",
]
