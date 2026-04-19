from __future__ import annotations

"""SGLang-inspired fused MoE draft for Diffulex.

This version keeps Diffulex's current high-level dispatcher contract
(`hidden_states`, `topk_ids`, `topk_weights`) but performs a local
expert-major pack/sort step before launching Triton kernels.
"""

from math import prod

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from diffulex.moe.metadata import ExpertExecutionMetadata

try:
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size
except ImportError:
    sgl_moe_align_block_size = None


PACKED_BLOCK_M = 16
WORKSPACE_CACHE: dict[tuple[str, int, torch.dtype], torch.Tensor] = {}


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
    buffer = WORKSPACE_CACHE.get(key)
    if buffer is None or buffer.numel() < numel:
        buffer = torch.empty(numel, device=device, dtype=dtype)
        WORKSPACE_CACHE[key] = buffer
    out = buffer[:numel].view(*shape)
    if zero:
        out.zero_()
    elif fill_value is not None:
        out.fill_(fill_value)
    return out


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
    expert_ids_ptr,
    w_ptr,
    out_ptr,
    num_rows,
    num_cols,
    k_dim,
    stride_am,
    stride_ak,
    stride_t,
    stride_e,
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
    expert_ids = tl.load(
        expert_ids_ptr + offs_m * stride_e,
        mask=row_mask,
        other=0,
    ).to(tl.int32)
    valid_rows = row_mask & (token_ids >= 0)

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
            + expert_ids[:, None, None] * stride_we
            + offs_n[None, :, None] * stride_wn
            + current_k[None, None, :] * stride_wk,
            mask=valid_rows[:, None, None] & (offs_n[None, :, None] < num_cols) & k_mask[None, None, :],
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
    expert_ids_ptr,
    w_ptr,
    out_ptr,
    num_rows,
    num_cols,
    k_dim,
    stride_am,
    stride_ak,
    stride_e,
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
    expert_ids = tl.load(
        expert_ids_ptr + offs_m * stride_e,
        mask=row_mask,
        other=0,
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
            + expert_ids[:, None, None] * stride_we
            + offs_n[None, :, None] * stride_wn
            + current_k[None, None, :] * stride_wk,
            mask=row_mask[:, None, None] & (offs_n[None, :, None] < num_cols) & k_mask[None, None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(a[:, None, :] * b, axis=2)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=row_mask[:, None] & (offs_n[None, :] < num_cols),
    )


@triton.jit
def _grouped_expert_gemm_packed_aligned(
    a_ptr,
    sorted_slot_ids_ptr,
    expert_block_ids_ptr,
    w_ptr,
    out_ptr,
    num_slots,
    num_padded_slots,
    num_cols,
    k_dim,
    num_local_experts,
    stride_am,
    stride_ak,
    stride_s,
    stride_b,
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
    row_mask = offs_m < num_padded_slots

    slot_ids = tl.load(
        sorted_slot_ids_ptr + offs_m * stride_s,
        mask=row_mask,
        other=num_slots,
    ).to(tl.int32)
    expert_id = tl.load(
        expert_block_ids_ptr + pid_m * stride_b,
        mask=pid_m >= 0,
        other=-1,
    ).to(tl.int32)
    valid_rows = row_mask & (slot_ids >= 0) & (slot_ids < num_slots) & (expert_id >= 0) & (expert_id < num_local_experts)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    for k_start in range(0, k_dim, BLOCK_K):
        current_k = k_start + k_offsets
        k_mask = current_k < k_dim
        a = tl.load(
            a_ptr + slot_ids[:, None] * stride_am + current_k[None, :] * stride_ak,
            mask=valid_rows[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            w_ptr
            + expert_id * stride_we
            + offs_n[:, None] * stride_wn
            + current_k[None, :] * stride_wk,
            mask=(expert_id >= 0) & (expert_id < num_local_experts) & (offs_n[:, None] < num_cols) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a, tl.trans(b))

    tl.store(
        out_ptr + slot_ids[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=valid_rows[:, None] & (offs_n[None, :] < num_cols),
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
) -> ExpertExecutionMetadata:
    num_tokens, hidden_size = hidden_states.shape
    del hidden_size
    top_k = topk_ids.shape[1]
    num_slots = num_tokens * top_k

    if num_slots == 0:
        empty_i32 = torch.empty((0,), device=hidden_states.device, dtype=torch.int32)
        empty_w = torch.empty((0,), device=hidden_states.device, dtype=topk_weights.dtype)
        return ExpertExecutionMetadata(
            packed_token_ids=empty_i32,
            packed_local_expert_ids=empty_i32,
            packed_weights=empty_w,
            num_slots=0,
        )

    local_ids = topk_ids.to(torch.int32) - int(local_expert_start)
    local_mask = (local_ids >= 0) & (local_ids < num_local_experts)
    flat_valid_mask = local_mask.reshape(-1)

    flat_local_ids = local_ids.reshape(-1)
    flat_token_ids = (
        torch.arange(num_tokens, device=hidden_states.device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(num_tokens, top_k)
        .reshape(-1)
    )
    flat_weights = topk_weights.reshape(-1)

    packed_token_ids = _get_workspace_tensor(
        "packed_token_ids",
        (num_slots,),
        dtype=torch.int32,
        device=hidden_states.device,
    )
    packed_local_expert_ids = _get_workspace_tensor(
        "packed_local_expert_ids",
        (num_slots,),
        dtype=torch.int32,
        device=hidden_states.device,
    )
    packed_weights = _get_workspace_tensor(
        "packed_weights",
        (num_slots,),
        dtype=topk_weights.dtype,
        device=hidden_states.device,
    )
    packed_token_ids.copy_(torch.where(flat_valid_mask, flat_token_ids, torch.full_like(flat_token_ids, -1)))
    packed_local_expert_ids.copy_(
        torch.where(flat_valid_mask, flat_local_ids, torch.zeros_like(flat_local_ids))
    )
    packed_weights.copy_(
        torch.where(flat_valid_mask, flat_weights, torch.zeros_like(flat_weights))
    )

    return ExpertExecutionMetadata(
        packed_token_ids=packed_token_ids,
        packed_local_expert_ids=packed_local_expert_ids,
        packed_weights=packed_weights,
        num_slots=num_slots,
    )


def _build_aligned_block_metadata(
    packed_local_expert_ids: torch.Tensor,
    *,
    block_size: int,
    num_local_experts: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int | None]:
    if packed_local_expert_ids.numel() == 0 or sgl_moe_align_block_size is None:
        return None, None, None
    topk_ids = packed_local_expert_ids[:, None].contiguous()
    num_slots = int(topk_ids.numel())
    max_num_tokens_padded = num_slots + (num_local_experts + 1) * (block_size - 1)
    sorted_slot_ids = torch.empty(
        (max_num_tokens_padded,),
        dtype=torch.int32,
        device=packed_local_expert_ids.device,
    )
    expert_block_ids = torch.empty(
        (triton.cdiv(max_num_tokens_padded, block_size),),
        dtype=torch.int32,
        device=packed_local_expert_ids.device,
    )
    num_tokens_post_pad = torch.empty(
        (1,),
        dtype=torch.int32,
        device=packed_local_expert_ids.device,
    )
    cumsum_buffer = torch.empty(
        (num_local_experts + 2,),
        dtype=torch.int32,
        device=packed_local_expert_ids.device,
    )
    sgl_moe_align_block_size(
        topk_ids,
        num_local_experts + 1,
        block_size,
        sorted_slot_ids,
        expert_block_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    total = int(num_tokens_post_pad.item())
    num_blocks = triton.cdiv(total, block_size)
    return sorted_slot_ids[:total].contiguous(), expert_block_ids[:num_blocks].contiguous(), total


def _launch_grouped_expert_gemm_gathered(
    hidden_states: torch.Tensor,
    w: torch.Tensor,
    packed_inputs: ExpertExecutionMetadata,
    *,
    num_cols: int,
    k_dim: int,
) -> torch.Tensor:
    packed_out = _get_workspace_tensor(
        "gate_up",
        (packed_inputs.num_slots, num_cols),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    if packed_inputs.num_slots == 0:
        return packed_out

    block_m = PACKED_BLOCK_M
    block_n = 64 if num_cols >= 64 else triton.next_power_of_2(num_cols)
    block_k = 32 if k_dim >= 32 else triton.next_power_of_2(k_dim)
    _grouped_expert_gemm_gathered[
        (triton.cdiv(packed_inputs.num_slots, block_m), triton.cdiv(num_cols, block_n))
    ](
        hidden_states,
        packed_inputs.packed_token_ids,
        packed_inputs.packed_local_expert_ids,
        w,
        packed_out,
        packed_inputs.num_slots,
        num_cols,
        k_dim,
        hidden_states.stride(0),
        hidden_states.stride(1),
        packed_inputs.packed_token_ids.stride(0),
        packed_inputs.packed_local_expert_ids.stride(0),
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
    packed_inputs: ExpertExecutionMetadata,
    *,
    num_cols: int,
    k_dim: int,
    workspace_name: str = "packed_slot_outputs",
) -> torch.Tensor:
    packed_out = _get_workspace_tensor(
        workspace_name,
        (packed_inputs.num_slots, num_cols),
        dtype=packed_hidden_states.dtype,
        device=packed_hidden_states.device,
    )
    if packed_inputs.num_slots == 0:
        return packed_out

    block_m = PACKED_BLOCK_M
    block_n = 64 if num_cols >= 64 else triton.next_power_of_2(num_cols)
    block_k = 32 if k_dim >= 32 else triton.next_power_of_2(k_dim)
    if (
        packed_inputs.sorted_slot_ids is not None
        and packed_inputs.expert_block_ids is not None
        and packed_inputs.num_tokens_post_padded is not None
    ):
        _grouped_expert_gemm_packed_aligned[
            (triton.cdiv(packed_inputs.num_tokens_post_padded, block_m), triton.cdiv(num_cols, block_n))
        ](
            packed_hidden_states,
            packed_inputs.sorted_slot_ids,
            packed_inputs.expert_block_ids,
            w,
            packed_out,
            packed_inputs.num_slots,
            packed_inputs.num_tokens_post_padded,
            num_cols,
            k_dim,
            w.shape[0],
            packed_hidden_states.stride(0),
            packed_hidden_states.stride(1),
            packed_inputs.sorted_slot_ids.stride(0),
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

    _grouped_expert_gemm_packed[
        (triton.cdiv(packed_inputs.num_slots, block_m), triton.cdiv(num_cols, block_n))
    ](
        packed_hidden_states,
        packed_inputs.packed_local_expert_ids,
        w,
        packed_out,
        packed_inputs.num_slots,
        num_cols,
        k_dim,
        packed_hidden_states.stride(0),
        packed_hidden_states.stride(1),
        packed_inputs.packed_local_expert_ids.stride(0),
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
    packed_inputs: ExpertExecutionMetadata,
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
    if packed_inputs.num_slots == 0:
        return combined_output.to(packed_slot_outputs.dtype)

    block_m = PACKED_BLOCK_M
    block_n = 64 if packed_slot_outputs.shape[1] >= 64 else triton.next_power_of_2(packed_slot_outputs.shape[1])
    _weighted_scatter_add[
        (triton.cdiv(packed_inputs.num_slots, block_m), triton.cdiv(packed_slot_outputs.shape[1], block_n))
    ](
        packed_slot_outputs,
        packed_inputs.packed_token_ids,
        packed_inputs.packed_weights,
        combined_output,
        packed_inputs.num_slots,
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
    )

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
) -> torch.Tensor:
    return _launch_fused_moe_kernels(
        hidden_states=hidden_states,
        w13=w13,
        w2=w2,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        local_expert_start=local_expert_start,
        hidden_act=hidden_act,
    )


def fused_expert_packed(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    execution_metadata: ExpertExecutionMetadata,
    *,
    hidden_act: str = "silu",
) -> torch.Tensor:
    hidden_states = hidden_states.contiguous()
    if hidden_states.shape[0] == 0:
        return hidden_states.new_empty((0, w2.shape[1]))
    _validate_fused_moe_inputs(
        hidden_states=hidden_states,
        w13=w13,
        w2=w2,
        topk_ids=execution_metadata.packed_local_expert_ids[:, None],
        topk_weights=execution_metadata.packed_weights[:, None],
        hidden_act=hidden_act,
    )
    num_local_experts = w13.shape[0]
    if execution_metadata.disable_aligned_metadata:
        sorted_slot_ids = None
        expert_block_ids = None
        num_tokens_post_padded = None
    else:
        sorted_slot_ids, expert_block_ids, num_tokens_post_padded = _build_aligned_block_metadata(
            execution_metadata.packed_local_expert_ids,
            block_size=PACKED_BLOCK_M,
            num_local_experts=num_local_experts,
        )
    packed_inputs = ExpertExecutionMetadata(
        packed_token_ids=execution_metadata.packed_token_ids,
        packed_local_expert_ids=execution_metadata.packed_local_expert_ids,
        packed_weights=execution_metadata.packed_weights,
        num_slots=execution_metadata.num_slots,
        seg_indptr=execution_metadata.seg_indptr,
        num_recv_tokens_per_expert=execution_metadata.num_recv_tokens_per_expert,
        sorted_slot_ids=sorted_slot_ids,
        expert_block_ids=expert_block_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        disable_aligned_metadata=execution_metadata.disable_aligned_metadata,
    )
    hidden_size = hidden_states.shape[1]
    intermediate_twice = w13.shape[1]
    intermediate_size = intermediate_twice // 2
    gate_up = _launch_grouped_expert_gemm_packed(
        hidden_states,
        w13,
        packed_inputs,
        num_cols=intermediate_twice,
        k_dim=hidden_size,
        workspace_name="packed_gate_up",
    )
    gate = gate_up[:, :intermediate_size]
    up = gate_up[:, intermediate_size:]
    F.silu(gate, inplace=True)
    gate.mul_(up)
    return _launch_grouped_expert_gemm_packed(
        gate,
        w2,
        packed_inputs,
        num_cols=w2.shape[1],
        k_dim=intermediate_size,
        workspace_name="packed_slot_outputs",
    ).to(hidden_states.dtype)
