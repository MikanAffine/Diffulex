import triton

from diffulex_kernel.python.chunked_prefill_triton import _prune_chunked_prefill_configs


def _config(block_n: int) -> triton.Config:
    return triton.Config({"BLOCK_M": 64, "BLOCK_N": block_n}, num_warps=4, num_stages=2)


def test_prune_chunked_prefill_configs_keeps_stable_block_n_for_small_blocks() -> None:
    configs = [_config(64), _config(128)]

    pruned = _prune_chunked_prefill_configs(configs, {}, DLLM_BLOCK_SIZE=16)

    assert [config.kwargs["BLOCK_N"] for config in pruned] == [64]


def test_prune_chunked_prefill_configs_keeps_search_space_for_large_blocks() -> None:
    configs = [_config(64), _config(128)]

    pruned = _prune_chunked_prefill_configs(configs, {}, DLLM_BLOCK_SIZE=32)

    assert [config.kwargs["BLOCK_N"] for config in pruned] == [64, 128]
