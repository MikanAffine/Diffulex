from __future__ import annotations

import torch

import diffulex_kernel.python.vllm_fuse_moe as vllm_moe


def test_fused_experts_last_chunk_keeps_topk_cache_rows(monkeypatch):
    monkeypatch.setattr(vllm_moe.envs, "VLLM_FUSED_MOE_CHUNK_SIZE", 4, raising=False)
    monkeypatch.setattr(vllm_moe.envs, "VLLM_MOE_DP_CHUNK_SIZE", 4, raising=False)
    monkeypatch.setattr(
        vllm_moe,
        "try_get_optimal_moe_config",
        lambda *args, **kwargs: {
            "BLOCK_SIZE_M": 4,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 8,
            "GROUP_SIZE_M": 1,
        },
    )

    def fake_align(topk_ids, block_size, num_experts):
        slots = topk_ids.numel()
        return (
            torch.arange(slots, dtype=torch.int32),
            torch.zeros(max(1, (slots + block_size - 1) // block_size), dtype=torch.int32),
            torch.tensor(slots, dtype=torch.int32),
        )

    monkeypatch.setattr(vllm_moe, "moe_align_block_size", fake_align)
    monkeypatch.setattr(
        vllm_moe,
        "invoke_fused_moe_kernel",
        lambda curr_hidden_states, w, out, *args, **kwargs: out.zero_(),
    )

    seen_silu_shapes = []

    def fake_silu_and_mul(out, x):
        seen_silu_shapes.append((tuple(out.shape), tuple(x.shape)))
        assert out.shape[0] == x.shape[0]
        out.zero_()

    def fake_moe_sum(x, out):
        assert x.shape[0] == out.shape[0]
        out.zero_()

    monkeypatch.setattr(torch.ops._C, "silu_and_mul", fake_silu_and_mul)
    monkeypatch.setattr(vllm_moe.ops, "moe_sum", fake_moe_sum)

    hidden_states = torch.zeros((6, 8), dtype=torch.bfloat16)
    w1 = torch.zeros((4, 16, 8), dtype=torch.bfloat16)
    w2 = torch.zeros((4, 8, 8), dtype=torch.bfloat16)
    topk_weights = torch.ones((6, 2), dtype=torch.float32)
    topk_ids = torch.zeros((6, 2), dtype=torch.int64)

    out = vllm_moe.fused_experts_impl(hidden_states, w1, w2, topk_weights, topk_ids)

    assert out.shape == hidden_states.shape
    assert seen_silu_shapes == [((8, 8), (8, 16)), ((4, 8), (4, 16))]
