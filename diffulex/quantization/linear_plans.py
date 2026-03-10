"""
Linear forward execution plans for quantization.

Plans are small, stateful callables that fix the dispatch path (bf16 / int8 / GPTQ / AWQ / Marlin)
for CUDA-graph friendly and low-overhead forward. Built by linear_plan_builder from layer state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class ForwardPlanSig:
    """Signature for validating cached forward plans (CUDA-graph friendly, no device sync)."""

    device_type: str
    device_index: int
    x_dtype: torch.dtype
    x_shape: tuple[int, ...]
    has_bias: bool
    mode: str  # "bf16" | "quant" | "offline"
    strategy_name: str


class ForwardPlanBase:
    """Base for all linear forward plans."""

    sig: ForwardPlanSig

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BF16Plan(ForwardPlanBase):
    def __init__(
        self,
        *,
        sig: ForwardPlanSig,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> None:
        self.sig = sig
        self._weight = weight
        self._bias = bias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._weight, self._bias)


class QuantInt8W8A16Plan(ForwardPlanBase):
    def __init__(
        self,
        *,
        sig: ForwardPlanSig,
        strategy,
        quant_kind: str,
        qweight: torch.Tensor,
        scales_1xn: torch.Tensor,
        out_features: int,
        bias: Optional[torch.Tensor],
    ) -> None:
        self.sig = sig
        self._strategy = strategy
        self._quant_kind = (quant_kind or "other").strip().lower() or "other"
        self._qweight = qweight
        self._scales_1xn = scales_1xn
        self._out_features = int(out_features)
        self._bias = bias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._strategy.linear_forward(
            x,
            self._qweight,
            self._bias,
            quant_kind=self._quant_kind,
            quant_scales=self._scales_1xn,
            out_features=self._out_features,
        )


class QuantInt8W8A8Plan(ForwardPlanBase):
    def __init__(
        self,
        *,
        sig: ForwardPlanSig,
        strategy,
        quant_kind: str,
        qweight: torch.Tensor,
        scales_1xn: torch.Tensor,
        out_features: int,
        bias: Optional[torch.Tensor],
    ) -> None:
        self.sig = sig
        self._strategy = strategy
        self._quant_kind = (quant_kind or "other").strip().lower() or "other"
        self._qweight = qweight
        self._scales_1xn = scales_1xn
        self._out_features = int(out_features)
        self._bias = bias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._strategy.linear_forward(
            x,
            self._qweight,
            self._bias,
            quant_kind=self._quant_kind,
            quant_scales=self._scales_1xn,
            out_features=self._out_features,
        )


class QuantGenericPlan(ForwardPlanBase):
    def __init__(
        self,
        *,
        sig: ForwardPlanSig,
        strategy,
        quant_kind: str,
        weight: torch.Tensor,
        scales: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> None:
        self.sig = sig
        self._strategy = strategy
        self._quant_kind = (quant_kind or "other").strip().lower() or "other"
        self._weight = weight
        self._scales = scales
        self._bias = bias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._strategy.linear_forward(
            x,
            self._weight,
            self._bias,
            quant_kind=self._quant_kind,
            quant_scales=self._scales,
        )


class OfflineGPTQPlan(ForwardPlanBase):
    def __init__(
        self,
        *,
        sig: ForwardPlanSig,
        strategy,
        quant_kind: str,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
        weight_bits: int,
        out_features: int,
        in_features: int,
        group_size: int,
        bias: Optional[torch.Tensor],
    ) -> None:
        self.sig = sig
        self._strategy = strategy
        self._quant_kind = (quant_kind or "other").strip().lower() or "other"
        self._qweight = qweight
        self._qzeros = qzeros
        self._scales = scales
        self._g_idx = g_idx
        self._weight_bits = int(weight_bits)
        self._out_features = int(out_features)
        self._in_features = int(in_features)
        self._group_size = int(group_size)
        self._bias = bias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._strategy.linear_forward(
            x,
            None,
            self._bias,
            quant_kind=self._quant_kind,
            gptq_qweight=self._qweight,
            gptq_qzeros=self._qzeros,
            gptq_scales=self._scales,
            gptq_g_idx=self._g_idx,
            weight_bits=self._weight_bits,
            use_v2_format=False,
            out_features=self._out_features,
            in_features=self._in_features,
            group_size=self._group_size,
        )


class OfflineAWQPlan(ForwardPlanBase):
    def __init__(
        self,
        *,
        sig: ForwardPlanSig,
        strategy,
        quant_kind: str,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        pack_factor: int,
        out_features: int,
        in_features: int,
        group_size: int,
        bias: Optional[torch.Tensor],
    ) -> None:
        self.sig = sig
        self._strategy = strategy
        self._quant_kind = (quant_kind or "other").strip().lower() or "other"
        self._qweight = qweight
        self._qzeros = qzeros
        self._scales = scales
        self._pack_factor = int(pack_factor)
        self._out_features = int(out_features)
        self._in_features = int(in_features)
        self._group_size = int(group_size)
        self._bias = bias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._strategy.linear_forward(
            x,
            None,
            self._bias,
            quant_kind=self._quant_kind,
            awq_qweight=self._qweight,
            awq_qzeros=self._qzeros,
            awq_scales=self._scales,
            pack_factor=self._pack_factor,
            out_features=self._out_features,
            in_features=self._in_features,
            group_size=self._group_size,
        )


class OfflineGPTQMarlinPlan(ForwardPlanBase):
    def __init__(
        self,
        *,
        sig: ForwardPlanSig,
        strategy,
        quant_kind: str,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zp: torch.Tensor,
        g_idx: torch.Tensor,
        g_idx_sort_indices: torch.Tensor,
        workspace: torch.Tensor,
        in_features: int,
        out_features: int,
        group_size: int,
        weight_bits: int,
        tp_dim: Optional[int],
        bias: Optional[torch.Tensor],
    ) -> None:
        self.sig = sig
        self._strategy = strategy
        self._quant_kind = (quant_kind or "other").strip().lower() or "other"
        self._qweight = qweight
        self._scales = scales
        self._zp = zp
        self._g_idx = g_idx
        self._g_idx_sort_indices = g_idx_sort_indices
        self._workspace = workspace
        self._in_features = int(in_features)
        self._out_features = int(out_features)
        self._group_size = int(group_size)
        self._weight_bits = int(weight_bits)
        self._tp_dim = tp_dim
        self._bias = bias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._strategy.linear_forward(
            x,
            None,
            self._bias,
            quant_kind=self._quant_kind,
            qweight=self._qweight,
            scales=self._scales,
            zp=self._zp,
            g_idx=self._g_idx,
            g_idx_sort_indices=self._g_idx_sort_indices,
            workspace=self._workspace,
            in_features=self._in_features,
            out_features=self._out_features,
            group_size=self._group_size,
            weight_bits=self._weight_bits,
            tp_dim=self._tp_dim,
        )


class OfflineAWQMarlinPlan(ForwardPlanBase):
    def __init__(
        self,
        *,
        sig: ForwardPlanSig,
        strategy,
        quant_kind: str,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zp: torch.Tensor,
        workspace: torch.Tensor,
        in_features: int,
        out_features: int,
        group_size: int,
        tp_dim: Optional[int],
        bias: Optional[torch.Tensor],
    ) -> None:
        self.sig = sig
        self._strategy = strategy
        self._quant_kind = (quant_kind or "other").strip().lower() or "other"
        self._qweight = qweight
        self._scales = scales
        self._zp = zp
        self._workspace = workspace
        self._in_features = int(in_features)
        self._out_features = int(out_features)
        self._group_size = int(group_size)
        self._tp_dim = tp_dim
        self._bias = bias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._strategy.linear_forward(
            x,
            None,
            self._bias,
            quant_kind=self._quant_kind,
            qweight=self._qweight,
            scales=self._scales,
            zp=self._zp,
            workspace=self._workspace,
            in_features=self._in_features,
            out_features=self._out_features,
            group_size=self._group_size,
            tp_dim=self._tp_dim,
        )


class DirectGPTQGemmPlan(ForwardPlanBase):
    """Direct GPTQ GEMM via torch.ops._C.gptq_gemm."""

    def __init__(
        self,
        *,
        sig: ForwardPlanSig,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
        weight_bits: int,
        out_features: int,
        bias: Optional[torch.Tensor],
        use_exllama: bool = True,
        use_v2_format: bool = False,
        cast_back_to_x_dtype: bool = True,
    ) -> None:
        self.sig = sig
        self._qweight = qweight
        self._qzeros = qzeros
        self._scales = scales
        self._g_idx = g_idx
        self._weight_bits = int(weight_bits)
        self._out_features = int(out_features)
        self._bias = bias
        self._use_exllama = bool(use_exllama)
        self._use_v2_format = bool(use_v2_format)
        self._cast_back = bool(cast_back_to_x_dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x if x.dtype == torch.float16 else x.to(dtype=torch.float16)
        x2 = x_in.reshape(-1, x_in.shape[-1]) if x_in.dim() != 2 else x_in
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        out = torch.ops._C.gptq_gemm(
            x2,
            self._qweight,
            self._qzeros,
            self._scales,
            self._g_idx,
            self._use_exllama,
            self._use_v2_format,
            self._weight_bits,
        )
        if self._bias is not None:
            out.add_(self._bias.to(dtype=out.dtype))
        out = out.reshape(x.shape[:-1] + (self._out_features,))
        if self._cast_back and out.dtype != x.dtype:
            return out.to(dtype=x.dtype)
        return out


class DirectAWQGemmPlan(ForwardPlanBase):
    """Direct AWQ GEMM (bypass Python strategy)."""

    def __init__(
        self,
        *,
        sig: ForwardPlanSig,
        awq_gemm,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        out_features: int,
        bias: Optional[torch.Tensor],
        split_k_iters: int = 1,
        cast_back_to_x_dtype: bool = True,
    ) -> None:
        self.sig = sig
        self._awq_gemm = awq_gemm
        self._qweight = qweight
        self._qzeros = qzeros
        self._scales = scales
        self._out_features = int(out_features)
        self._bias = bias
        self._split_k_iters = int(split_k_iters)
        self._cast_back = bool(cast_back_to_x_dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x if x.dtype == torch.float16 else x.to(dtype=torch.float16)
        reshaped_x = x_in.reshape(-1, x_in.shape[-1])
        if not reshaped_x.is_contiguous():
            reshaped_x = reshaped_x.contiguous()
        out = self._awq_gemm(reshaped_x, self._qweight, self._scales, self._qzeros, self._split_k_iters)
        if self._bias is not None:
            out.add_(self._bias.to(dtype=out.dtype))
        out = out.reshape(x.shape[:-1] + (self._out_features,))
        if self._cast_back and out.dtype != x.dtype:
            return out.to(dtype=x.dtype)
        return out


class DirectMarlinGemmPlan(ForwardPlanBase):
    """Direct Marlin GEMM via torch.ops._C.gptq_marlin_gemm."""

    def __init__(
        self,
        *,
        sig: ForwardPlanSig,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zp: torch.Tensor,
        g_idx: torch.Tensor,
        g_idx_sort_indices: torch.Tensor,
        workspace: torch.Tensor,
        wtype_id: int,
        n: int,
        is_k_full: bool,
        use_atomic_add: bool,
        marlin_bias: Optional[torch.Tensor],
        cast_back_to_x_dtype: bool = True,
    ) -> None:
        self.sig = sig
        self._qweight = qweight
        self._scales = scales
        self._zp = zp
        self._g_idx = g_idx
        self._g_idx_sort_indices = g_idx_sort_indices
        self._workspace = workspace
        self._wtype_id = int(wtype_id)
        self._n = int(n)
        self._is_k_full = bool(is_k_full)
        self._use_atomic_add = bool(use_atomic_add)
        self._bias = marlin_bias
        self._cast_back = bool(cast_back_to_x_dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (int(self._n),)
        m = int(reshaped_x.shape[0])
        k = int(reshaped_x.shape[1])
        out = torch.ops._C.gptq_marlin_gemm(
            reshaped_x,
            None,
            self._qweight,
            self._bias,
            self._scales,
            None,
            None,
            self._zp,
            self._g_idx,
            self._g_idx_sort_indices,
            self._workspace,
            self._wtype_id,
            m,
            int(self._n),
            k,
            self._is_k_full,
            self._use_atomic_add,
            True,
            False,
        )
        out = out.reshape(out_shape)
        if self._cast_back and out.dtype != x.dtype:
            return out.to(dtype=x.dtype)
        return out


__all__ = [
    "ForwardPlanSig",
    "ForwardPlanBase",
    "BF16Plan",
    "QuantInt8W8A16Plan",
    "QuantInt8W8A8Plan",
    "QuantGenericPlan",
    "OfflineGPTQPlan",
    "OfflineAWQPlan",
    "OfflineGPTQMarlinPlan",
    "OfflineAWQMarlinPlan",
    "DirectGPTQGemmPlan",
    "DirectAWQGemmPlan",
    "DirectMarlinGemmPlan",
]
