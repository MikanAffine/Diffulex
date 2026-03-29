from types import SimpleNamespace

from diffulex.utils.output import GenerationOutputs


def test_generation_outputs_handles_empty_prefill_suffix() -> None:
    outputs = GenerationOutputs(1)
    req = SimpleNamespace(
        req_id=0,
        is_prefilling=True,
        new_tokens=0,
        running_sequence=[],
        block_size=4,
        dllm_block_buffer=SimpleNamespace(dllm_blocks=[]),
        truncated_response=[],
        full_response=[],
        is_truncated=False,
        max_new_tokens_reached=False,
        max_model_len_reached=False,
        eos_token_generated=False,
    )

    outputs.record_step([req], step_time=1.0)

    assert outputs.prefill_throughput == 0
    assert outputs.postfix()["ptps"] == "0tok/sec"


def test_generation_outputs_decode_throughput_uses_batch_time() -> None:
    outputs = GenerationOutputs(2)
    shared_buffer = SimpleNamespace(dllm_blocks=[])
    reqs = [
        SimpleNamespace(
            req_id=0,
            is_prefilling=False,
            new_tokens=1,
            running_sequence=[1, 2, 3, 4],
            block_size=4,
            dllm_block_buffer=shared_buffer,
            truncated_response=[],
            full_response=[],
            is_truncated=False,
            max_new_tokens_reached=False,
            max_model_len_reached=False,
            eos_token_generated=False,
        ),
        SimpleNamespace(
            req_id=1,
            is_prefilling=False,
            new_tokens=1,
            running_sequence=[5, 6, 7, 8],
            block_size=4,
            dllm_block_buffer=shared_buffer,
            truncated_response=[],
            full_response=[],
            is_truncated=False,
            max_new_tokens_reached=False,
            max_model_len_reached=False,
            eos_token_generated=False,
        ),
    ]

    outputs.record_step(reqs, step_time=1.0)

    assert outputs.decode_throughput == 2.0
    assert outputs.tpf == 2.0
    assert outputs.total_time == 1.0
    assert outputs.postfix()["dtps"] == "2tok/sec"


def test_generation_outputs_prefill_throughput_uses_batch_time() -> None:
    outputs = GenerationOutputs(2)
    shared_buffer = SimpleNamespace(dllm_blocks=[])
    reqs = [
        SimpleNamespace(
            req_id=0,
            is_prefilling=True,
            new_tokens=0,
            running_sequence=[1, 2, 3, 4],
            block_size=4,
            dllm_block_buffer=shared_buffer,
            truncated_response=[],
            full_response=[],
            is_truncated=False,
            max_new_tokens_reached=False,
            max_model_len_reached=False,
            eos_token_generated=False,
        ),
        SimpleNamespace(
            req_id=1,
            is_prefilling=True,
            new_tokens=0,
            running_sequence=[5, 6, 7, 8],
            block_size=4,
            dllm_block_buffer=shared_buffer,
            truncated_response=[],
            full_response=[],
            is_truncated=False,
            max_new_tokens_reached=False,
            max_model_len_reached=False,
            eos_token_generated=False,
        ),
    ]

    outputs.record_step(reqs, step_time=2.0)

    assert outputs.prefill_throughput == 4.0
    assert outputs.total_time == 2.0
    assert outputs.postfix()["ptps"] == "4tok/sec"
