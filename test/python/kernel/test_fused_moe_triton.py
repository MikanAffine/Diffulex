import pytest
import torch

from diffulex.moe.metadata import ExpertExecutionMetadata
from diffulex_kernel.python.fused_moe_triton import (
    PACKED_BLOCK_M,
    _build_aligned_block_metadata_triton,
    fused_expert_packed,
    fused_moe,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_moe_is_cuda_graph_capture_safe() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16

    num_tokens = 8
    hidden_size = 16
    intermediate_size = 32
    num_local_experts = 2
    top_k = 2
    local_expert_start = 1

    hidden_states = torch.randn((num_tokens, hidden_size), device=device, dtype=dtype)
    w13 = torch.randn((num_local_experts, intermediate_size * 2, hidden_size), device=device, dtype=dtype)
    w2 = torch.randn((num_local_experts, hidden_size, intermediate_size), device=device, dtype=dtype)
    topk_ids = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [0, 3],
            [1, 1],
            [2, 2],
            [3, 0],
            [1, 3],
        ],
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.rand((num_tokens, top_k), device=device, dtype=dtype)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    eager_out = fused_moe(
        hidden_states,
        w13,
        w2,
        topk_ids,
        topk_weights,
        local_expert_start=local_expert_start,
    )

    static_hidden_states = hidden_states.clone()
    static_topk_ids = topk_ids.clone()
    static_topk_weights = topk_weights.clone()
    graph_out = torch.empty_like(eager_out)

    # Warm up workspace allocation before capture.
    fused_moe(
        static_hidden_states,
        w13,
        w2,
        static_topk_ids,
        static_topk_weights,
        local_expert_start=local_expert_start,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out.copy_(
            fused_moe(
                static_hidden_states,
                w13,
                w2,
                static_topk_ids,
                static_topk_weights,
                local_expert_start=local_expert_start,
            )
        )

    graph.replay()

    torch.testing.assert_close(graph_out, eager_out, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_moe_align_block_metadata_groups_slots_by_expert() -> None:
    device = torch.device("cuda")
    expert_ids = torch.tensor([1, 0, 1, 2, 0, 1, 2], device=device, dtype=torch.int32)

    sorted_slot_ids, expert_block_ids, num_padded = _build_aligned_block_metadata_triton(
        expert_ids,
        block_size=4,
        num_local_experts=3,
    )
    torch.cuda.synchronize()

    assert num_padded == expert_ids.numel() + 4 * (3 + 1 - 1)
    assert sorted_slot_ids[:4].cpu().tolist() == [1, 4, -1, -1]
    assert sorted_slot_ids[4:8].cpu().tolist() == [0, 2, 5, -1]
    assert sorted_slot_ids[8:12].cpu().tolist() == [3, 6, -1, -1]
    assert expert_block_ids[:3].cpu().tolist() == [0, 1, 2]
    assert torch.all(expert_block_ids[3:] == -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_expert_packed_triton_aligned_metadata_matches_unaligned() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16

    num_slots = 11
    hidden_size = 16
    intermediate_size = 32
    num_local_experts = 3

    hidden_states = torch.randn((num_slots, hidden_size), device=device, dtype=dtype)
    w13 = torch.randn((num_local_experts, intermediate_size * 2, hidden_size), device=device, dtype=dtype)
    w2 = torch.randn((num_local_experts, hidden_size, intermediate_size), device=device, dtype=dtype)
    packed_local_expert_ids = torch.tensor(
        [0, 2, 1, 1, 0, 2, 2, 1, 0, 1, 2],
        device=device,
        dtype=torch.int32,
    )
    packed_token_ids = torch.arange(num_slots, device=device, dtype=torch.int32)
    packed_weights = torch.ones((num_slots,), device=device, dtype=dtype)

    unaligned_metadata = ExpertExecutionMetadata(
        packed_token_ids=packed_token_ids,
        packed_local_expert_ids=packed_local_expert_ids,
        packed_weights=packed_weights,
        num_slots=num_slots,
        disable_aligned_metadata=True,
    )
    aligned_sorted_slot_ids, aligned_expert_block_ids, num_tokens_post_padded = (
        _build_aligned_block_metadata_triton(
            packed_local_expert_ids,
            block_size=PACKED_BLOCK_M,
            num_local_experts=num_local_experts,
        )
    )
    aligned_metadata = ExpertExecutionMetadata(
        packed_token_ids=packed_token_ids,
        packed_local_expert_ids=packed_local_expert_ids,
        packed_weights=packed_weights,
        num_slots=num_slots,
        sorted_slot_ids=aligned_sorted_slot_ids,
        expert_block_ids=aligned_expert_block_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )

    unaligned = fused_expert_packed(hidden_states, w13, w2, unaligned_metadata)
    aligned = fused_expert_packed(hidden_states, w13, w2, aligned_metadata)

    torch.testing.assert_close(aligned, unaligned, rtol=1e-3, atol=1e-3)
