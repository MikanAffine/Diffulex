import pytest
import torch

from diffulex_kernel.python.fused_moe_triton import fused_moe


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
