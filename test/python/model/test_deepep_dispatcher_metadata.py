from __future__ import annotations

import torch

from diffulex.moe.dispatcher.deepep_dispatcher import DeepEPDispatcher
from diffulex.moe.metadata import DispatcherStage
import diffulex.moe.dispatcher.deepep_dispatcher as deepep_dispatcher_module


def test_deepep_dispatch_b_expands_native_recv_tokens_to_expert_major_slots():
    dispatcher = DeepEPDispatcher.__new__(DeepEPDispatcher)
    dispatcher._stage = DispatcherStage.AFTER_DISPATCH_A
    dispatcher._dispatch_intermediate_state = None
    dispatcher.top_k = 2
    dispatcher.num_local_experts = 2
    dispatcher.local_expert_start = 2
    dispatcher.local_expert_end = 4

    original_hidden_states = torch.empty((5, 2), dtype=torch.bfloat16)
    native_recv_hidden_states = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
        dtype=torch.bfloat16,
    )
    native_recv_topk_ids = torch.tensor(
        [
            [2, 0],
            [3, 2],
            [1, 3],
        ],
        dtype=torch.int64,
    )
    native_recv_topk_weights = torch.tensor(
        [
            [0.1, 0.9],
            [0.2, 0.3],
            [0.4, 0.5],
        ],
        dtype=torch.float32,
    )
    handle = object()
    dispatcher._dispatch_intermediate_state = (
        original_hidden_states,
        True,
        5,
        native_recv_hidden_states,
        native_recv_topk_ids,
        native_recv_topk_weights,
        handle,
        torch.tensor([1, 2], dtype=torch.int32),
    )

    output = dispatcher.dispatch_b()
    metadata = output.metadata

    assert metadata.native_handle is handle
    assert metadata.native_recv_num_tokens == 3
    assert metadata.total_recv_slots == 4
    assert torch.equal(metadata.recv_local_expert, torch.tensor([0, 0, 1, 1], dtype=torch.int32))
    assert torch.equal(metadata.recv_token_indices, torch.tensor([0, 1, 1, 2], dtype=torch.int32))
    assert torch.allclose(metadata.recv_weights, torch.tensor([0.1, 0.3, 0.2, 0.5]))
    assert torch.equal(
        output.recv_hidden_states.float(),
        torch.tensor([[10.0, 11.0], [20.0, 21.0], [20.0, 21.0], [30.0, 31.0]]),
    )
    assert torch.equal(metadata.seg_indptr, torch.tensor([0, 2, 4]))
    assert torch.equal(metadata.num_recv_tokens_per_expert, torch.tensor([2, 2], dtype=torch.int32))

    expert_major_values = torch.arange(4)
    assert torch.equal(
        DeepEPDispatcher._restore_src_order(expert_major_values, metadata),
        torch.tensor([0, 2, 1, 3]),
    )


def test_deepep_dispatcher_protocol_with_mock_native_buffer(monkeypatch):
    class FakeEvent:
        def current_stream_wait(self):
            raise AssertionError("async_finish=False should not wait fake event")

    class FakeNativeBuffer:
        @staticmethod
        def get_dispatch_config(_ep_size):
            return object()

        @staticmethod
        def get_combine_config(_ep_size):
            return object()

        def get_dispatch_layout(self, topk_ids, num_experts, **_kwargs):
            assert num_experts == 4
            assert torch.equal(topk_ids, torch.tensor([[0, 2], [1, 3]], dtype=torch.int64))
            num_tokens_per_rank = torch.tensor([2, 2], dtype=torch.int32)
            num_tokens_per_expert = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
            is_token_in_rank = torch.tensor([[True, True], [True, True]])
            return num_tokens_per_rank, None, num_tokens_per_expert, is_token_in_rank, FakeEvent()

        def dispatch(self, hidden_states, *, topk_idx, topk_weights, **_kwargs):
            assert hidden_states.dtype == torch.bfloat16
            assert topk_weights.dtype == torch.float32
            return hidden_states, topk_idx, topk_weights, [1, 1], "handle", FakeEvent()

        def combine(self, token_outputs, handle, **_kwargs):
            assert handle == "handle"
            self.combined_token_outputs = token_outputs
            return token_outputs, None, FakeEvent()

    fake_buffer = FakeNativeBuffer()
    monkeypatch.setattr(deepep_dispatcher_module, "DeepEPBuffer", FakeNativeBuffer)

    dispatcher = DeepEPDispatcher.__new__(DeepEPDispatcher)
    dispatcher._stage = DispatcherStage.INITIAL
    dispatcher._dispatch_intermediate_state = None
    dispatcher._combine_intermediate_state = None
    dispatcher._get_buffer = lambda: fake_buffer
    dispatcher.ep_size = 2
    dispatcher.num_experts = 4
    dispatcher.num_local_experts = 2
    dispatcher.local_expert_start = 0
    dispatcher.local_expert_end = 2
    dispatcher.top_k = 2
    dispatcher.async_finish = False
    dispatcher.deepep_mode = "normal"

    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.bfloat16)

    dispatched = dispatcher.dispatch(hidden_states, topk_ids, topk_weights)
    assert dispatched.metadata.native_handle == "handle"
    assert dispatched.metadata.native_recv_num_tokens == 2
    assert dispatched.metadata.total_recv_slots == 2
    assert torch.equal(dispatched.recv_local_expert_ids, torch.tensor([0, 1], dtype=torch.int32))
    assert torch.equal(dispatched.metadata.recv_token_indices, torch.tensor([0, 1], dtype=torch.int32))
    assert torch.allclose(dispatched.recv_weights, torch.tensor([0.25, 0.5]))

    slot_outputs = torch.tensor([[10.0, 1.0], [2.0, 20.0]], dtype=torch.bfloat16)
    combined = dispatcher.combine(slot_outputs, dispatched.metadata)
    assert torch.equal(combined, fake_buffer.combined_token_outputs)
    assert torch.allclose(combined.float(), torch.tensor([[10.0, 1.0], [2.0, 20.0]]))


def test_deepep_low_latency_protocol_with_mock_native_buffer(monkeypatch):
    class FakeEvent:
        def current_stream_wait(self):
            return None

    class FakeNativeBuffer:
        @staticmethod
        def get_dispatch_config(_ep_size):
            return object()

        @staticmethod
        def get_combine_config(_ep_size):
            return object()

        @staticmethod
        def get_low_latency_rdma_size_hint(*_args):
            return 1024

        def low_latency_dispatch(
            self,
            hidden_states,
            topk_ids,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            **_kwargs,
        ):
            assert num_max_dispatch_tokens_per_rank == 4
            assert num_experts == 4
            recv = torch.zeros((2, 4, 2), dtype=torch.bfloat16)
            recv[0, 0] = hidden_states[0]
            recv[1, 0] = hidden_states[1]
            recv_count = torch.tensor([1, 1], dtype=torch.int32)
            return recv, recv_count, "ll_handle", FakeEvent(), None

        def low_latency_combine(self, packed_outputs, topk_ids, topk_weights, handle, **_kwargs):
            assert handle == "ll_handle"
            self.packed_outputs = packed_outputs
            combined = torch.stack((packed_outputs[0, 0], packed_outputs[1, 0]))
            return combined, FakeEvent(), None

    fake_buffer = FakeNativeBuffer()
    monkeypatch.setattr(deepep_dispatcher_module, "DeepEPBuffer", FakeNativeBuffer)

    dispatcher = DeepEPDispatcher.__new__(DeepEPDispatcher)
    dispatcher._stage = DispatcherStage.INITIAL
    dispatcher._dispatch_intermediate_state = None
    dispatcher._combine_intermediate_state = None
    dispatcher._get_buffer = lambda: fake_buffer
    dispatcher.ep_size = 2
    dispatcher.num_experts = 4
    dispatcher.num_local_experts = 2
    dispatcher.local_expert_start = 0
    dispatcher.local_expert_end = 2
    dispatcher.top_k = 2
    dispatcher.async_finish = False
    dispatcher.deepep_mode = "low_latency"
    dispatcher.num_max_dispatch_tokens_per_rank = 4

    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float32)

    dispatched = dispatcher.dispatch(hidden_states, topk_ids, topk_weights)
    metadata = dispatched.metadata
    assert metadata.low_latency
    assert metadata.total_recv_slots == 8
    assert torch.equal(dispatched.recv_local_expert_ids, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32))
    assert torch.equal(metadata.num_recv_tokens_per_expert, torch.tensor([1, 1], dtype=torch.int32))
    assert torch.equal(metadata.seg_indptr, torch.tensor([0, 4, 8]))

    slot_outputs = torch.arange(16, dtype=torch.bfloat16).reshape(8, 2)
    combined = dispatcher.combine(slot_outputs, metadata)
    assert fake_buffer.packed_outputs.shape == (2, 4, 2)
    assert torch.equal(combined, torch.stack((fake_buffer.packed_outputs[0, 0], fake_buffer.packed_outputs[1, 0])))
