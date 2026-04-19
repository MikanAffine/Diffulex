from __future__ import annotations

import pytest

import diffulex.distributed.parallel_state as parallel_state


def test_get_world_size_supports_global_ep_topology():
    assert parallel_state.get_world_size(tp_size=4, ep_size=32, dp_size=8) == 32
    assert parallel_state.get_world_size(tp_size=4, ep_size=4, dp_size=8) == 32
    assert parallel_state.get_world_size(tp_size=1, ep_size=8, dp_size=8) == 8


def test_get_world_size_rejects_unsupported_topology():
    with pytest.raises(NotImplementedError):
        parallel_state.get_world_size(tp_size=4, ep_size=16, dp_size=8)


def test_build_parallel_state_for_test_global_ep_rank_metadata():
    state = parallel_state.build_parallel_state_for_test(
        tp_size=4,
        ep_size=32,
        dp_size=8,
        global_rank=9,
    )

    assert state.topology == "global_ep"
    assert state.world_size == 32
    assert state.dp_rank == 2
    assert state.tp_rank == 1
    assert state.base_model.tp_ranks == (8, 9, 10, 11)
    assert state.base_model.dp_ranks == (1, 5, 9, 13, 17, 21, 25, 29)
    assert state.ep_rank == 9
    assert state.moe is not None
    assert state.moe.ep_ranks == tuple(range(32))


def test_build_parallel_state_for_test_per_dp_ep_rank_metadata():
    state = parallel_state.build_parallel_state_for_test(
        tp_size=4,
        ep_size=4,
        dp_size=8,
        global_rank=9,
    )

    assert state.topology == "ep_per_dp_shard"
    assert state.world_size == 32
    assert state.dp_rank == 2
    assert state.tp_rank == 1
    assert state.base_model.tp_ranks == (8, 9, 10, 11)
    assert state.moe is not None
    assert state.ep_rank == 1
    assert state.moe.ep_ranks == (8, 9, 10, 11)
