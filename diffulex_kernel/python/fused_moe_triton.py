from __future__ import annotations

"""SGLang-inspired fused MoE draft for Diffulex.

This version keeps Diffulex's current high-level dispatcher contract
(`hidden_states`, `topk_ids`, `topk_weights`) but performs a local
expert-major pack/sort step before launching Triton kernels.
"""

from dataclasses import dataclass
from math import prod

import torch
import torch.nn.functional as F

import triton
import triton.language as tl


PACKED_BLOCK_M = 16
_WORKSPACE_CACHE: dict[tuple[str, int, torch.dtype], torch.Tensor] = {}


def _workspace_device_index(device: torch.device) -> int:
    return device.index if device.index is not None else -1


def _get_workspace_tensor(
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    zero: bool = False,
    fill_value: int | float | None = None,
) -> torch.Tensor:
    numel = prod(shape)
    key = (name, _workspace_device_index(device), dtype)
    buffer = _WORKSPACE_CACHE.get(key)
    if buffer is None or buffer.numel() < numel:
        buffer = torch.empty(numel, device=device, dtype=dtype)
        _WORKSPACE_CACHE[key] = buffer
    out = buffer[:numel].view(*shape)
    if zero:
        out.zero_()
    elif fill_value is not None:
        out.fill_(fill_value)
    return out


@dataclass
class PackedMoEInputs:
    packed_token_ids: torch.Tensor
    packed_weights: torch.Tensor
    expert_block_ids: torch.Tensor
    num_valid_slots: int
    num_padded_slots: int
    single_slot_per_token: bool


def _validate_fused_moe_inputs(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden_act: str,
) -> None:
    if hidden_act != "silu":
        raise ValueError(f"Only silu experts are supported right now, got {hidden_act!r}.")
    if not hidden_states.is_cuda:
        raise ValueError("fused_moe requires CUDA tensors.")
    if hidden_states.dim() != 2 or topk_ids.dim() != 2 or topk_weights.dim() != 2:
        raise ValueError(
            "fused_moe expects hidden_states/topk_ids/topk_weights to be 2D tensors."
        )
    if hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            f"fused_moe only supports fp16/bf16/fp32 activations, got {hidden_states.dtype}."
        )
    if w13.dtype != hidden_states.dtype or w2.dtype != hidden_states.dtype:
        raise TypeError(
            "fused_moe expects hidden_states, w13, and w2 to share the same dtype."
        )
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            f"topk_ids and topk_weights must have the same shape, got {topk_ids.shape} and {topk_weights.shape}."
        )
    if w13.shape[0] != w2.shape[0] or w13.shape[2] != hidden_states.shape[1]:
        raise ValueError(
            "Weight shapes do not match hidden_states or local expert count."
        )
    if w13.shape[1] % 2 != 0:
        raise ValueError("fused_moe expects w13.shape[1] to be 2 * intermediate_size.")


@triton.jit
def _grouped_expert_gemm_gathered(
    a_ptr,
    token_ids_ptr,
    expert_block_ids_ptr,
    w_ptr,
    out_ptr,
    num_rows,
    num_cols,
    k_dim,
    stride_am,
    stride_ak,
    stride_t,
    stride_be,
    stride_we,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = offs_m < num_rows

    token_ids = tl.load(
        token_ids_ptr + offs_m * stride_t,
        mask=row_mask,
        other=-1,
    ).to(tl.int32)
    valid_rows = row_mask & (token_ids >= 0)
    expert_id = tl.load(
        expert_block_ids_ptr + pid_m * stride_be,
    ).to(tl.int32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    for k_start in range(0, k_dim, BLOCK_K):
        current_k = k_start + k_offsets
        k_mask = current_k < k_dim
        a = tl.load(
            a_ptr + token_ids[:, None] * stride_am + current_k[None, :] * stride_ak,
            mask=valid_rows[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            w_ptr
            + expert_id * stride_we
            + offs_n[None, :, None] * stride_wn
            + current_k[None, None, :] * stride_wk,
            mask=(offs_n[None, :, None] < num_cols) & k_mask[None, None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(a[:, None, :] * b, axis=2)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=row_mask[:, None] & (offs_n[None, :] < num_cols),
    )


@triton.jit
def _grouped_expert_gemm_packed(
    a_ptr,
    expert_block_ids_ptr,
    w_ptr,
    out_ptr,
    num_rows,
    num_cols,
    k_dim,
    stride_am,
    stride_ak,
    stride_be,
    stride_we,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = offs_m < num_rows
    expert_id = tl.load(
        expert_block_ids_ptr + pid_m * stride_be,
    ).to(tl.int32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    for k_start in range(0, k_dim, BLOCK_K):
        current_k = k_start + k_offsets
        k_mask = current_k < k_dim
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + current_k[None, :] * stride_ak,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            w_ptr
            + expert_id * stride_we
            + offs_n[None, :, None] * stride_wn
            + current_k[None, None, :] * stride_wk,
            mask=(offs_n[None, :, None] < num_cols) & k_mask[None, None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(a[:, None, :] * b, axis=2)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=row_mask[:, None] & (offs_n[None, :] < num_cols),
    )


@triton.jit
def _weighted_scatter_add(
    slot_outputs_ptr,
    token_ids_ptr,
    weights_ptr,
    out_ptr,
    num_rows,
    num_cols,
    stride_sm,
    stride_sn,
    stride_t,
    stride_w,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = offs_m < num_rows
    col_mask = offs_n < num_cols

    token_ids = tl.load(
        token_ids_ptr + offs_m * stride_t,
        mask=row_mask,
        other=-1,
    ).to(tl.int64)
    valid_rows = row_mask & (token_ids >= 0)
    weights = tl.load(
        weights_ptr + offs_m * stride_w,
        mask=valid_rows,
        other=0.0,
    ).to(tl.float32)
    values = tl.load(
        slot_outputs_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn,
        mask=valid_rows[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    weighted = values * weights[:, None]

    out_ptrs = out_ptr + token_ids[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.atomic_add(
        out_ptrs,
        weighted,
        mask=valid_rows[:, None] & col_mask[None, :],
    )


def _pack_fused_moe_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_local_experts: int,
    local_expert_start: int,
    block_m: int,
    topk_ids_are_local: bool,
) -> PackedMoEInputs:
    num_tokens, hidden_size = hidden_states.shape
    del hidden_size
    top_k = topk_ids.shape[1]
    device = hidden_states.device
    flat_token_ids = torch.arange(num_tokens * top_k, device=device, dtype=torch.int32)
    flat_token_ids = torch.div(flat_token_ids, top_k, rounding_mode="floor")
    flat_weights = topk_weights.reshape(-1)

    if topk_ids_are_local:
        valid_local_ids = topk_ids.to(torch.int32).reshape(-1)
        valid_token_ids = flat_token_ids
        valid_weights = flat_weights
    else:
        flat_local_ids = topk_ids.to(torch.int32).reshape(-1) - int(local_expert_start)
        flat_valid_mask = (flat_local_ids >= 0) & (flat_local_ids < num_local_experts)
        valid_local_ids = flat_local_ids[flat_valid_mask]
        valid_token_ids = flat_token_ids[flat_valid_mask]
        valid_weights = flat_weights[flat_valid_mask]

    num_valid_slots = valid_local_ids.numel()
    if num_valid_slots == 0:
        empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)
        empty_w = torch.empty((0,), device=device, dtype=topk_weights.dtype)
        return PackedMoEInputs(
            packed_token_ids=empty_i32,
            packed_weights=empty_w,
            expert_block_ids=empty_i32,
            num_valid_slots=0,
            num_padded_slots=0,
            single_slot_per_token=(top_k == 1),
        )

    sorted_local_ids, sort_order = torch.sort(valid_local_ids)
    sorted_token_ids = valid_token_ids[sort_order]
    sorted_weights = valid_weights[sort_order]

    counts = torch.bincount(sorted_local_ids.to(torch.int64), minlength=num_local_experts)
    padded_counts = ((counts + block_m - 1) // block_m) * block_m
    num_padded_slots = int(padded_counts.sum().item())
    num_expert_blocks = num_padded_slots // block_m

    packed_token_ids = _get_workspace_tensor(
        "packed_token_ids",
        (num_padded_slots,),
        dtype=torch.int32,
        device=device,
        fill_value=-1,
    )
    packed_weights = _get_workspace_tensor(
        "packed_weights",
        (num_padded_slots,),
        dtype=topk_weights.dtype,
        device=device,
        zero=True,
    )
    expert_block_ids = _get_workspace_tensor(
        "expert_block_ids",
        (num_expert_blocks,),
        dtype=torch.int32,
        device=device,
    )
    offsets = _get_workspace_tensor(
        "expert_offsets",
        (num_local_experts,),
        dtype=torch.int64,
        device=device,
    )
    dst_positions = _get_workspace_tensor(
        "dst_positions",
        (num_valid_slots,),
        dtype=torch.int64,
        device=device,
    )

    offsets.copy_(torch.cumsum(padded_counts.to(torch.int64), dim=0) - padded_counts.to(torch.int64))
    expert_block_ids.copy_(
        torch.repeat_interleave(
            torch.arange(num_local_experts, device=device, dtype=torch.int32),
            (padded_counts // block_m).to(torch.int64),
            output_size=num_expert_blocks,
        )
    )

    slot_indices = _get_workspace_tensor(
        "slot_indices",
        (num_valid_slots,),
        dtype=torch.int64,
        device=device,
    )
    slot_indices.copy_(torch.arange(num_valid_slots, device=device, dtype=torch.int64))
    segment_start_positions = _get_workspace_tensor(
        "segment_start_positions",
        (num_valid_slots,),
        dtype=torch.int64,
        device=device,
        zero=True,
    )
    segment_start_positions[0] = 0
    if num_valid_slots > 1:
        segment_start_positions[1:] = torch.where(
            sorted_local_ids[1:] != sorted_local_ids[:-1],
            slot_indices[1:],
            torch.zeros_like(slot_indices[1:]),
        )
    segment_start_positions = torch.cummax(segment_start_positions, dim=0).values
    dst_positions.copy_(slot_indices)
    dst_positions.sub_(segment_start_positions)
    dst_positions.add_(offsets[sorted_local_ids.to(torch.int64)])

    packed_token_ids[dst_positions] = sorted_token_ids
    packed_weights[dst_positions] = sorted_weights

    return PackedMoEInputs(
        packed_token_ids=packed_token_ids,
        packed_weights=packed_weights,
        expert_block_ids=expert_block_ids,
        num_valid_slots=num_valid_slots,
        num_padded_slots=num_padded_slots,
        single_slot_per_token=(top_k == 1),
    )


def _launch_grouped_expert_gemm_gathered(
    hidden_states: torch.Tensor,
    w: torch.Tensor,
    packed_inputs: PackedMoEInputs,
    *,
    num_cols: int,
    k_dim: int,
) -> torch.Tensor:
    packed_out = _get_workspace_tensor(
        "gate_up",
        (packed_inputs.num_padded_slots, num_cols),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    if packed_inputs.num_padded_slots == 0:
        return packed_out

    block_m = PACKED_BLOCK_M
    block_n = 64 if num_cols >= 64 else triton.next_power_of_2(num_cols)
    block_k = 32 if k_dim >= 32 else triton.next_power_of_2(k_dim)
    _grouped_expert_gemm_gathered[
        (triton.cdiv(packed_inputs.num_padded_slots, block_m), triton.cdiv(num_cols, block_n))
    ](
        hidden_states,
        packed_inputs.packed_token_ids,
        packed_inputs.expert_block_ids,
        w,
        packed_out,
        packed_inputs.num_padded_slots,
        num_cols,
        k_dim,
        hidden_states.stride(0),
        hidden_states.stride(1),
        packed_inputs.packed_token_ids.stride(0),
        packed_inputs.expert_block_ids.stride(0),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        packed_out.stride(0),
        packed_out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return packed_out


def _launch_grouped_expert_gemm_packed(
    packed_hidden_states: torch.Tensor,
    w: torch.Tensor,
    packed_inputs: PackedMoEInputs,
    *,
    num_cols: int,
    k_dim: int,
) -> torch.Tensor:
    packed_out = _get_workspace_tensor(
        "packed_slot_outputs",
        (packed_inputs.num_padded_slots, num_cols),
        dtype=packed_hidden_states.dtype,
        device=packed_hidden_states.device,
    )
    if packed_inputs.num_padded_slots == 0:
        return packed_out

    block_m = PACKED_BLOCK_M
    block_n = 64 if num_cols >= 64 else triton.next_power_of_2(num_cols)
    block_k = 32 if k_dim >= 32 else triton.next_power_of_2(k_dim)
    _grouped_expert_gemm_packed[
        (triton.cdiv(packed_inputs.num_padded_slots, block_m), triton.cdiv(num_cols, block_n))
    ](
        packed_hidden_states,
        packed_inputs.expert_block_ids,
        w,
        packed_out,
        packed_inputs.num_padded_slots,
        num_cols,
        k_dim,
        packed_hidden_states.stride(0),
        packed_hidden_states.stride(1),
        packed_inputs.expert_block_ids.stride(0),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        packed_out.stride(0),
        packed_out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return packed_out


def _combine_packed_moe_outputs(
    packed_slot_outputs: torch.Tensor,
    packed_inputs: PackedMoEInputs,
    *,
    num_tokens: int,
) -> torch.Tensor:
    combined_output = _get_workspace_tensor(
        "combined_output",
        (num_tokens, packed_slot_outputs.shape[1]),
        dtype=torch.float32,
        device=packed_slot_outputs.device,
        zero=True,
    )
    if packed_inputs.num_valid_slots == 0:
        return combined_output.to(packed_slot_outputs.dtype)

    if packed_inputs.single_slot_per_token:
        valid_rows = packed_inputs.packed_token_ids >= 0
        valid_token_ids = packed_inputs.packed_token_ids[valid_rows].to(torch.int64)
        valid_outputs = packed_slot_outputs[valid_rows].to(torch.float32)
        valid_weights = packed_inputs.packed_weights[valid_rows].to(torch.float32).unsqueeze(-1)
        combined_output[valid_token_ids] = valid_outputs * valid_weights
        return combined_output.to(packed_slot_outputs.dtype)

    block_m = PACKED_BLOCK_M
    block_n = 64 if packed_slot_outputs.shape[1] >= 64 else triton.next_power_of_2(packed_slot_outputs.shape[1])
    _weighted_scatter_add[
        (triton.cdiv(packed_inputs.num_padded_slots, block_m), triton.cdiv(packed_slot_outputs.shape[1], block_n))
    ](
        packed_slot_outputs,
        packed_inputs.packed_token_ids,
        packed_inputs.packed_weights,
        combined_output,
        packed_inputs.num_padded_slots,
        packed_slot_outputs.shape[1],
        packed_slot_outputs.stride(0),
        packed_slot_outputs.stride(1),
        packed_inputs.packed_token_ids.stride(0),
        packed_inputs.packed_weights.stride(0),
        combined_output.stride(0),
        combined_output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return combined_output.to(packed_slot_outputs.dtype)


def _launch_fused_moe_kernels(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    local_expert_start: int,
    hidden_act: str,
    topk_ids_are_local: bool,
) -> torch.Tensor:
    hidden_states = hidden_states.contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()

    if hidden_states.shape[0] == 0 or topk_ids.shape[1] == 0:
        return hidden_states.new_zeros((hidden_states.shape[0], w2.shape[1]))

    _validate_fused_moe_inputs(
        hidden_states=hidden_states,
        w13=w13,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_act=hidden_act,
    )

    num_tokens, hidden_size = hidden_states.shape
    num_local_experts = w13.shape[0]
    intermediate_twice = w13.shape[1]
    intermediate_size = intermediate_twice // 2

    packed_inputs = _pack_fused_moe_inputs(
        hidden_states,
        topk_ids,
        topk_weights,
        num_local_experts=num_local_experts,
        local_expert_start=local_expert_start,
        block_m=PACKED_BLOCK_M,
        topk_ids_are_local=topk_ids_are_local,
    )
    if packed_inputs.num_valid_slots == 0:
        return hidden_states.new_zeros((num_tokens, w2.shape[1]))

    gate_up = _launch_grouped_expert_gemm_gathered(
        hidden_states,
        w13,
        packed_inputs,
        num_cols=intermediate_twice,
        k_dim=hidden_size,
    )

    gate = gate_up[:, :intermediate_size]
    up = gate_up[:, intermediate_size:]
    F.silu(gate, inplace=True)
    gate.mul_(up)
    packed_slot_outputs = _launch_grouped_expert_gemm_packed(
        gate,
        w2,
        packed_inputs,
        num_cols=w2.shape[1],
        k_dim=intermediate_size,
    )
    return _combine_packed_moe_outputs(
        packed_slot_outputs,
        packed_inputs,
        num_tokens=num_tokens,
    ).to(hidden_states.dtype)


def fused_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    local_expert_start: int = 0,
    hidden_act: str = "silu",
    topk_ids_are_local: bool = False,
) -> torch.Tensor:
    return _launch_fused_moe_kernels(
        hidden_states=hidden_states,
        w13=w13,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        local_expert_start=local_expert_start,
        hidden_act=hidden_act,
        topk_ids_are_local=topk_ids_are_local,
    )
