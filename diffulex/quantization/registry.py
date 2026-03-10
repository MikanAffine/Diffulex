"""
Quantization registries for Diffulex.
"""

from __future__ import annotations

from typing import Callable, Dict

from diffulex.quantization.kv_cache_dtype import _normalize_kv_cache_dtype
from diffulex.quantization.strategy import (
    KVCacheQuantizationStrategy,
    LinearQuantizationStrategy,
)

KVCacheStrategyBuilder = Callable[[], KVCacheQuantizationStrategy]
_KV_CACHE_BUILDERS: Dict[str, KVCacheStrategyBuilder] = {}


def register_kv_cache_strategy(
    *dtype_aliases: str,
) -> Callable[[KVCacheStrategyBuilder], KVCacheStrategyBuilder]:
    def _decorator(builder: KVCacheStrategyBuilder) -> KVCacheStrategyBuilder:
        for alias in dtype_aliases:
            key = _normalize_kv_cache_dtype(alias)
            _KV_CACHE_BUILDERS[key] = builder
        return builder

    return _decorator


def create_kv_cache_strategy(kv_cache_dtype: str) -> KVCacheQuantizationStrategy:
    key = _normalize_kv_cache_dtype(kv_cache_dtype)
    builder = _KV_CACHE_BUILDERS.get(key)
    if builder is None:
        raise ValueError(
            f"Unsupported kv_cache_dtype={kv_cache_dtype!r} (normalized={key!r}). "
            f"Registered: {sorted(_KV_CACHE_BUILDERS.keys())}"
        )
    return builder()


def registered_kv_cache_dtypes() -> list[str]:
    return sorted(_KV_CACHE_BUILDERS.keys())


LinearStrategyBuilder = Callable[[], LinearQuantizationStrategy]
_LINEAR_BUILDERS: Dict[tuple[str, str], LinearStrategyBuilder] = {}


def _normalize_linear_dtype(dtype: str) -> str:
    s = (dtype or "").strip().lower()
    if s in {"__stub__", "__fallback__"}:
        return "__stub__"
    aliases = {
        "": "bf16",
        "none": "bf16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "int8": "int8",
        "i8": "int8",
        "int4": "int4",
        "i4": "int4",
        "fp8": "fp8_e4m3",
        "fp8_e4m3": "fp8_e4m3",
        "e4m3": "fp8_e4m3",
        "fp8_e5m2": "fp8_e5m2",
        "e5m2": "fp8_e5m2",
        "gptq": "gptq",
        "gptq_marlin": "gptq_marlin",
        "gptq_marlin_24": "gptq_marlin_24",
        "awq": "awq",
        "awq_marlin": "awq_marlin",
        "gptq_awq": "gptq_awq",
        "marlin": "marlin_int8",
        "marlin_int8": "marlin_int8",
    }
    if s not in aliases:
        raise ValueError(
            f"Unsupported linear quant dtype={dtype!r}. Supported: bf16/int8/int4/fp8/fp8_e4m3/fp8_e5m2/gptq/awq/marlin"
        )
    return aliases[s]


def register_linear_strategy(
    *,
    weight_dtype: str,
    act_dtype: str,
) -> Callable[[LinearStrategyBuilder], LinearStrategyBuilder]:
    w = _normalize_linear_dtype(weight_dtype)
    a = _normalize_linear_dtype(act_dtype)

    def _decorator(builder: LinearStrategyBuilder) -> LinearStrategyBuilder:
        _LINEAR_BUILDERS[(w, a)] = builder
        return builder

    return _decorator


def create_linear_strategy(*, weight_dtype: str, act_dtype: str) -> LinearQuantizationStrategy:
    w = _normalize_linear_dtype(weight_dtype)
    a = _normalize_linear_dtype(act_dtype)
    builder = _LINEAR_BUILDERS.get((w, a))
    if builder is not None:
        return builder()
    stub = _LINEAR_BUILDERS.get(("__stub__", "__stub__"))
    if stub is None:
        raise ValueError(
            f"Unsupported linear strategy pair (weight_dtype={weight_dtype!r}, act_dtype={act_dtype!r}) "
            f"(normalized={(w, a)!r}). Registered pairs: {sorted(_LINEAR_BUILDERS.keys())}"
        )
    s = stub()
    try:
        setattr(s, "weight_dtype", w)
        setattr(s, "act_dtype", a)
    except Exception:
        pass
    return s


def registered_linear_dtypes() -> list[str]:
    return [
        "bf16",
        "int8",
        "int4",
        "fp8_e4m3",
        "fp8_e5m2",
        "gptq",
        "gptq_marlin",
        "gptq_marlin_24",
        "awq",
        "awq_marlin",
        "gptq_awq",
        "marlin_int8",
    ]
