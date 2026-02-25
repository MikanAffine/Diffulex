"""
Quantization strategy interfaces.

This module defines abstract base classes for different types of quantization strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol

import torch
import torch.nn.functional as F


class _AttnMetaDataLike(Protocol):
    """A minimal protocol for attention metadata used by Diffulex runtime."""

    k_scale: Optional[torch.Tensor]
    v_scale: Optional[torch.Tensor]


class QuantizationStrategy(ABC):
    """Quantization strategy abstract base class."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

    @abstractmethod
    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        """Returns storage dtype and itemsize."""
        pass

    @abstractmethod
    def quantize(self, tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        """Quantize a tensor."""
        pass

    @abstractmethod
    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        """Dequantize a tensor."""
        pass

    @abstractmethod
    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        """Returns the shape of scale tensor."""
        pass

    def configure(self, *, diffulex_config: Any | None = None) -> None:
        """Optional hook to configure a strategy from Diffulex `Config`."""
        _ = diffulex_config
        return

    @property
    def requires_runtime_scales(self) -> bool:
        """Whether this strategy requires runtime scale tensors to be allocated/updated."""
        return False


class KVCacheQuantizationStrategy(QuantizationStrategy):
    """KV Cache quantization strategy interface (extended interface)."""

    @property
    def kv_cache_format(self) -> str:
        return "bf16"

    @abstractmethod
    def compute_scales(self, k: torch.Tensor, v: torch.Tensor,
                      num_kv_heads: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute quantization scales for K and V."""
        pass

    @abstractmethod
    def update_scales(self, k: torch.Tensor, v: torch.Tensor,
                     k_scale: Optional[torch.Tensor], v_scale: Optional[torch.Tensor],
                     num_kv_heads: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Update quantization scales (e.g., using running max strategy)."""
        pass

    def init_scales(self, num_kv_heads: int, device: torch.device) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Initialize quantization scales for K and V."""
        return None, None

    @property
    def requires_kv_cache_scales(self) -> bool:
        return self.requires_runtime_scales

    def maybe_set_attn_metadata_scales(
        self,
        attn_metadata: _AttnMetaDataLike,
        *,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
    ) -> None:
        """Populate `attn_metadata.k_scale/v_scale` when needed."""
        if not self.requires_kv_cache_scales:
            return
        if k_scale is None or v_scale is None:
            raise ValueError(
                f"{self.name} requires k_scale/v_scale but got "
                f"k_scale={k_scale is not None}, v_scale={v_scale is not None}"
            )
        attn_metadata.k_scale = k_scale
        attn_metadata.v_scale = v_scale

    def view_kv_cache_for_kernels(self, cache: torch.Tensor) -> torch.Tensor:
        """Return a view of cache suitable for kernel consumption."""
        return cache

    def quantize_kv_for_store(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize K/V for KV cache store (optional helper)."""
        raise NotImplementedError(f"{self.name} does not implement quantize_kv_for_store")


class WeightQuantizationStrategy(QuantizationStrategy):
    """Weight quantization strategy interface (for future extension)."""

    @abstractmethod
    def quantize_weight(self, weight: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        """Quantize model weights."""
        pass

    @abstractmethod
    def dequantize_weight(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        """Dequantize model weights."""
        pass


class LinearQuantizationStrategy(QuantizationStrategy):
    """Linear layer quantization strategy interface (weights + activations)."""

    @property
    def linear_weight_format(self) -> str:
        return "bf16"

    @property
    def linear_act_format(self) -> str:
        return "bf16"

    def quantize_weight_for_kernel(
        self,
        weight: torch.Tensor,
        *,
        device: torch.device | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Optionally quantize/pack weight for kernel consumption."""
        if device is not None:
            weight = weight.to(device=device)
        return weight, None

    def quantize_act_for_kernel(
        self,
        x: torch.Tensor,
        *,
        device: torch.device | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Optionally quantize activations for kernel consumption."""
        if device is not None:
            x = x.to(device=device)
        return x, None

    def linear_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        quant_kind: str,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute Linear output for a given kind."""
        _ = quant_kind, kwargs
        return F.linear(x, weight, bias)
