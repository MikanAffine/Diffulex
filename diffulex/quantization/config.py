"""
Quantization configuration objects for Diffulex.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class kv_cacheQuantConfig:
    """KV-cache quantization configuration."""

    dtype: str = "bf16"


@dataclass(frozen=True)
class WeightQuantConfig:
    """Weight quantization configuration (placeholder)."""

    method: str = "none"
    linear_attn_dtype: str = "bf16"
    linear_mlp_dtype: str = "bf16"


@dataclass(frozen=True)
class ActivationQuantConfig:
    """Activation quantization configuration (placeholder)."""

    linear_attn_dtype: str = "bf16"
    linear_mlp_dtype: str = "bf16"


@dataclass(frozen=True)
class QuantizationConfig:
    """Top-level quantization configuration for Diffulex."""

    kv_cache: kv_cacheQuantConfig = kv_cacheQuantConfig()
    weights: WeightQuantConfig = WeightQuantConfig()
    activations: ActivationQuantConfig = ActivationQuantConfig()

    @classmethod
    def from_diffulex_config(cls, config) -> "QuantizationConfig":
        kv_cache_dtype = getattr(config, "kv_cache_dtype", "bf16") or "bf16"
        linear_attn_weight_dtype = getattr(config, "linear_attn_weight_dtype", "bf16") or "bf16"
        linear_mlp_weight_dtype = getattr(config, "linear_mlp_weight_dtype", "bf16") or "bf16"
        linear_attn_act_dtype = getattr(config, "linear_attn_act_dtype", "bf16") or "bf16"
        linear_mlp_act_dtype = getattr(config, "linear_mlp_act_dtype", "bf16") or "bf16"
        return cls(
            kv_cache=kv_cacheQuantConfig(dtype=kv_cache_dtype),
            weights=WeightQuantConfig(
                linear_attn_dtype=linear_attn_weight_dtype,
                linear_mlp_dtype=linear_mlp_weight_dtype,
            ),
            activations=ActivationQuantConfig(
                linear_attn_dtype=linear_attn_act_dtype,
                linear_mlp_dtype=linear_mlp_act_dtype,
            ),
        )
