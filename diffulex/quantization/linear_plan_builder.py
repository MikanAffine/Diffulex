"""
Build a cached forward plan for a linear layer (bf16 / online quant / offline GPTQ/AWQ/Marlin).

The layer passes itself and (example_x, bias); the builder returns a ForwardPlanBase
to be stored on the layer. Layer remains responsible for _maybe_promote_weight_to_quantized_at_runtime
and for calling this builder.
"""

from __future__ import annotations

from typing import Optional

import torch

from diffulex.quantization.linear_plans import (
    ForwardPlanSig,
    ForwardPlanBase,
    BF16Plan,
    QuantInt8W8A16Plan,
    QuantInt8W8A8Plan,
    QuantGenericPlan,
    OfflineGPTQPlan,
    OfflineAWQPlan,
    OfflineGPTQMarlinPlan,
    OfflineAWQMarlinPlan,
    DirectGPTQGemmPlan,
    DirectAWQGemmPlan,
    DirectMarlinGemmPlan,
)


def build_forward_plan(
    layer,
    example_x: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Optional[ForwardPlanBase]:
    """Build and return a forward plan for the given layer. Caller assigns to layer._forward_plan."""
    strategy = layer._get_linear_strategy()
    device = example_x.device
    dev_idx = layer._device_index(device)
    has_bias = bias is not None
    strategy_name = getattr(strategy, "name", "") if strategy is not None else ""

    def sig(mode: str) -> ForwardPlanSig:
        return ForwardPlanSig(
            device_type=device.type,
            device_index=dev_idx,
            x_dtype=example_x.dtype,
            x_shape=tuple(int(x) for x in example_x.shape),
            has_bias=has_bias,
            mode=mode,
            strategy_name=strategy_name,
        )

    if layer.has_offline_quantized_weight():
        if strategy is None:
            raise RuntimeError("Offline quantized weight is present but no linear strategy is configured.")
        weight_format = getattr(strategy, "linear_weight_format", None)
        out_features, in_features, group_size = layer._offline_meta()
        s = sig("offline")

        if weight_format == "gptq":
            layer._maybe_prepare_offline_gptq(example_x)
            bits = layer._infer_gptq_weight_bits(in_features=in_features)
            g_idx = layer.gptq_g_idx
            if g_idx.device != device:
                g_idx = g_idx.to(device=device, dtype=torch.int32)
            if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "gptq_gemm"):
                return DirectGPTQGemmPlan(
                    sig=s,
                    qweight=layer.gptq_qweight,
                    qzeros=layer.gptq_qzeros,
                    scales=layer.gptq_scales,
                    g_idx=g_idx,
                    weight_bits=bits,
                    out_features=out_features,
                    bias=bias,
                    use_exllama=True,
                    use_v2_format=False,
                    cast_back_to_x_dtype=True,
                )
            return OfflineGPTQPlan(
                sig=s,
                strategy=strategy,
                quant_kind=layer.quant_kind,
                qweight=layer.gptq_qweight,
                qzeros=layer.gptq_qzeros,
                scales=layer.gptq_scales,
                g_idx=g_idx,
                weight_bits=bits,
                out_features=out_features,
                in_features=in_features,
                group_size=group_size,
                bias=bias,
            )

        if weight_format == "awq":
            bits = int(layer._offline_quant_bits_py) if int(layer._offline_quant_bits_py) > 0 else 4
            pack_factor = 32 // max(1, bits)
            awq_gemm = getattr(torch.ops._C, "awq_gemm", None) if hasattr(torch.ops, "_C") else None
            if awq_gemm is not None:
                return DirectAWQGemmPlan(
                    sig=s,
                    awq_gemm=awq_gemm,
                    qweight=layer.awq_qweight,
                    qzeros=layer.awq_qzeros,
                    scales=layer.awq_scales,
                    out_features=out_features,
                    bias=bias,
                    split_k_iters=1,
                    cast_back_to_x_dtype=True,
                )
            return OfflineAWQPlan(
                sig=s,
                strategy=strategy,
                quant_kind=layer.quant_kind,
                qweight=layer.awq_qweight,
                qzeros=layer.awq_qzeros,
                scales=layer.awq_scales,
                pack_factor=pack_factor,
                out_features=out_features,
                in_features=in_features,
                group_size=group_size,
                bias=bias,
            )

        if weight_format == "gptq_marlin":
            layer._maybe_prepare_offline_gptq_marlin(example_x)
            bits = layer._infer_gptq_weight_bits(in_features=in_features)
            if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "gptq_marlin_gemm"):
                try:
                    from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # type: ignore
                        marlin_is_k_full,
                        marlin_make_empty_g_idx,
                        should_use_atomic_add_reduce,
                        marlin_permute_bias,
                    )
                    from vllm.scalar_type import scalar_types  # type: ignore
                except Exception:
                    marlin_is_k_full = None
                    marlin_make_empty_g_idx = None
                    should_use_atomic_add_reduce = None
                    marlin_permute_bias = None
                    scalar_types = None

                if scalar_types is None:
                    return OfflineGPTQMarlinPlan(
                        sig=s,
                        strategy=strategy,
                        quant_kind=layer.quant_kind,
                        qweight=layer.gptq_marlin_qweight,
                        scales=layer.gptq_marlin_scales,
                        zp=layer.gptq_marlin_zp,
                        g_idx=layer.gptq_marlin_g_idx,
                        g_idx_sort_indices=layer.gptq_marlin_g_idx_sort_indices,
                        workspace=layer.gptq_marlin_workspace,
                        in_features=in_features,
                        out_features=out_features,
                        group_size=group_size,
                        weight_bits=bits,
                        tp_dim=layer.tp_dim,
                        bias=bias,
                    )
                _empty = (marlin_make_empty_g_idx(device) if marlin_make_empty_g_idx else
                          torch.empty((0,), device=device, dtype=torch.int32))
                g_idx = layer.gptq_marlin_g_idx if layer.gptq_marlin_g_idx.numel() > 0 else _empty
                g_idx_sort = (layer.gptq_marlin_g_idx_sort_indices
                              if layer.gptq_marlin_g_idx_sort_indices.numel() > 0 else _empty)
                row_parallel = bool(layer.tp_dim == 1)
                has_g_idx = bool(g_idx.numel() > 0)
                is_k_full = marlin_is_k_full(has_g_idx, row_parallel) if marlin_is_k_full else True
                marlin_bias = marlin_permute_bias(bias) if bias is not None and marlin_permute_bias else (bias if bias is not None else None)
                reshaped_x = example_x.reshape(-1, example_x.shape[-1])
                m, n, k = int(reshaped_x.shape[0]), int(out_features), int(reshaped_x.shape[1])
                use_atomic_add = bool(should_use_atomic_add_reduce(m=m, n=n, k=k, device=device, dtype=reshaped_x.dtype)) if should_use_atomic_add_reduce else False
                wtype = scalar_types.uint4b8 if bits == 4 else (scalar_types.uint8b128 if bits == 8 else None)
                if wtype is None:
                    raise RuntimeError(f"gptq_marlin: unsupported weight_bits={bits} (expected 4 or 8)")
                return DirectMarlinGemmPlan(
                    sig=s,
                    qweight=layer.gptq_marlin_qweight,
                    scales=layer.gptq_marlin_scales,
                    zp=layer.gptq_marlin_zp,
                    g_idx=g_idx,
                    g_idx_sort_indices=g_idx_sort,
                    workspace=layer.gptq_marlin_workspace,
                    wtype_id=wtype.id,
                    n=out_features,
                    is_k_full=is_k_full,
                    use_atomic_add=use_atomic_add,
                    marlin_bias=marlin_bias,
                    cast_back_to_x_dtype=True,
                )
            return OfflineGPTQMarlinPlan(
                sig=s,
                strategy=strategy,
                quant_kind=layer.quant_kind,
                qweight=layer.gptq_marlin_qweight,
                scales=layer.gptq_marlin_scales,
                zp=layer.gptq_marlin_zp,
                g_idx=layer.gptq_marlin_g_idx,
                g_idx_sort_indices=layer.gptq_marlin_g_idx_sort_indices,
                workspace=layer.gptq_marlin_workspace,
                in_features=in_features,
                out_features=out_features,
                group_size=group_size,
                weight_bits=bits,
                tp_dim=layer.tp_dim,
                bias=bias,
            )

        if weight_format == "awq_marlin":
            layer._maybe_prepare_offline_awq_marlin(example_x)
            if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "gptq_marlin_gemm"):
                try:
                    from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # type: ignore
                        marlin_make_empty_g_idx,
                        should_use_atomic_add_reduce,
                        marlin_permute_bias,
                    )
                    from vllm.scalar_type import scalar_types  # type: ignore
                except Exception:
                    marlin_make_empty_g_idx = None
                    should_use_atomic_add_reduce = None
                    marlin_permute_bias = None
                    scalar_types = None
                if scalar_types is None:
                    return OfflineAWQMarlinPlan(
                        sig=s,
                        strategy=strategy,
                        quant_kind=layer.quant_kind,
                        qweight=layer.awq_marlin_qweight,
                        scales=layer.awq_marlin_scales,
                        zp=layer.awq_marlin_zp,
                        workspace=layer.awq_marlin_workspace,
                        in_features=in_features,
                        out_features=out_features,
                        group_size=group_size,
                        tp_dim=layer.tp_dim,
                        bias=bias,
                    )
                empty = marlin_make_empty_g_idx(device) if marlin_make_empty_g_idx else torch.empty((0,), device=device, dtype=torch.int32)
                marlin_bias = marlin_permute_bias(bias) if bias is not None and marlin_permute_bias else (bias if bias is not None else None)
                reshaped_x = example_x.reshape(-1, example_x.shape[-1])
                m, n, k = int(reshaped_x.shape[0]), int(out_features), int(reshaped_x.shape[1])
                use_atomic_add = bool(should_use_atomic_add_reduce(m=m, n=n, k=k, device=device, dtype=reshaped_x.dtype)) if should_use_atomic_add_reduce else False
                return DirectMarlinGemmPlan(
                    sig=s,
                    qweight=layer.awq_marlin_qweight,
                    scales=layer.awq_marlin_scales,
                    zp=layer.awq_marlin_zp,
                    g_idx=empty,
                    g_idx_sort_indices=empty,
                    workspace=layer.awq_marlin_workspace,
                    wtype_id=scalar_types.uint4.id,
                    n=out_features,
                    is_k_full=True,
                    use_atomic_add=use_atomic_add,
                    marlin_bias=marlin_bias,
                    cast_back_to_x_dtype=True,
                )
            return OfflineAWQMarlinPlan(
                sig=s,
                strategy=strategy,
                quant_kind=layer.quant_kind,
                qweight=layer.awq_marlin_qweight,
                scales=layer.awq_marlin_scales,
                zp=layer.awq_marlin_zp,
                workspace=layer.awq_marlin_workspace,
                in_features=in_features,
                out_features=out_features,
                group_size=group_size,
                tp_dim=layer.tp_dim,
                bias=bias,
            )

        raise RuntimeError(
            f"Offline quantized weight is present but strategy weight_format={weight_format!r} is not supported by forward plan."
        )

    if layer.has_quantized_weight():
        if strategy is None:
            raise RuntimeError("Quantized weight is present but no linear strategy is configured.")
        s = sig("quant")
        if getattr(strategy, "name", "") == "linear_int8_w8a16":
            return QuantInt8W8A16Plan(
                sig=s,
                strategy=strategy,
                quant_kind=layer.quant_kind,
                qweight=layer.quant_weight_int8,
                scales_1xn=layer.quant_scales_1xn,
                out_features=layer._forward_out_features,
                bias=bias,
            )
        if getattr(strategy, "name", "") == "linear_int8_w8a8":
            return QuantInt8W8A8Plan(
                sig=s,
                strategy=strategy,
                quant_kind=layer.quant_kind,
                qweight=layer.quant_weight_int8,
                scales_1xn=layer.quant_scales_1xn,
                out_features=layer._forward_out_features,
                bias=bias,
            )
        return QuantGenericPlan(
            sig=s,
            strategy=strategy,
            quant_kind=layer.quant_kind,
            weight=layer.quant_weight_int8,
            scales=layer.quant_scales,
            bias=bias,
        )

    weight = getattr(layer, "weight", None)
    if weight is None:
        raise RuntimeError("No quantized/offline weights are present but bf16 weight is missing.")
    return BF16Plan(sig=sig("bf16"), weight=weight, bias=bias)


__all__ = ["build_forward_plan"]
