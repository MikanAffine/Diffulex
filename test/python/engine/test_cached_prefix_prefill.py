from types import SimpleNamespace

import torch

from diffulex.mixin.multi_block.engine.model_runner import ModelRunnerMultiBlockMixin
from diffulex.sampler.sdar import SDARSampler


class _Runner(ModelRunnerMultiBlockMixin):
    page_size = 4
    block_size = 4


def test_prepare_prefill_req_uses_suffix_positions_and_lengths_for_cached_prefix() -> None:
    req = SimpleNamespace(
        running_sequence=list(range(8, 20)),
        in_cache_len=8,
        running_len=20,
        page_table=[0, 1, 2, 3, 4],
        prefix_len=20,
        padded_prefix_len=20,
        dllm_blocks=[
            SimpleNamespace(start=0, end=4, rel_page_id=0, is_to_cache=False),
            SimpleNamespace(start=4, end=8, rel_page_id=1, is_to_cache=False),
            SimpleNamespace(start=8, end=12, rel_page_id=2, is_to_cache=True),
            SimpleNamespace(start=12, end=16, rel_page_id=3, is_to_cache=True),
            SimpleNamespace(start=16, end=20, rel_page_id=4, is_to_cache=True),
        ],
    )

    prepared = _Runner()._prepare_prefill_req(req)

    assert prepared["input_ids"] == list(range(8, 20))
    assert prepared["positions"] == list(range(8, 20))
    assert prepared["context_len"] == 8
    assert prepared["seqlen_q"] == 12
    assert prepared["seqlen_k"] == 12
    assert prepared["valid_slice"] == 12
    assert prepared["slot_mapping"] == list(range(8, 20))


def test_sampler_prefill_localizes_mask_token_indices_after_cached_prefix() -> None:
    sampler = SDARSampler()
    sampler.fetch_attn_metadata = lambda: SimpleNamespace(is_prefill=[True])

    block = SimpleNamespace(
        block_id=1,
        is_active=True,
        num_mask_tokens=4,
        mask_token_global_ids=[4, 5, 6, 7],
        mask_token_relative_ids=[0, 1, 2, 3],
        should_force_decode_topk=False,
    )
    req = SimpleNamespace(
        req_id=0,
        running_sequence=[-1, -1, -1, -1],
        chunk_size=4,
        dllm_blocks=[block],
        in_cache_len=4,
    )

    logits = torch.zeros(4, 8)
    logits[:, 0] = 1.0
    temperatures = torch.tensor([0.0])

    out = sampler([req], logits, temperatures, threshold=0.95)

    assert out.sampled_tokens_map["0"]["0"] == [0, 0, 0, 0]
    assert out.mask_token_rel_ids_map["0"]["0"] == [0, 1, 2, 3]
