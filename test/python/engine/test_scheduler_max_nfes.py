from collections import deque
import json
from types import SimpleNamespace

import pytest

from diffulex.engine.status import DllmReqStatus
from diffulex.strategy_template.multi_block.engine.request import MultiBlockReqTemplate
from diffulex.strategy_template.multi_block.engine.scheduler import MultiBlockSchedulerTemplate


class _Scheduler(MultiBlockSchedulerTemplate):
    def __init__(self):
        self.kv_cache_manager = SimpleNamespace(free=self._free)
        self.running_reqs = deque()
        self.freed_req_ids: list[int] = []

    def _free(self, req):
        self.freed_req_ids.append(req.req_id)

    def add(self, req):
        pass

    def schedule(self):
        pass

    def preempt(self, req):
        pass

    def postprocess(self, reqs, sample_output):
        pass


class _Req:
    def __init__(self, req_id: int, max_nfe: int | None = None, max_repetition_run: int | None = None):
        self.req_id = req_id
        self.max_nfe = max_nfe
        self.max_repetition_run = max_repetition_run
        self.nfe = 0
        self.new_tokens = 0
        self.dllm_blocks = []
        self.status = DllmReqStatus.DECODING
        self.repetition_run_length = 0
        self.eos_token_generated = False
        self.max_new_tokens_reached = False
        self.max_model_len_reached = False
        self.truncated_response = []
        self.completion_reason = None
        self.token_ids = []

    @property
    def max_nfe_reached(self) -> bool:
        return self.max_nfe is not None and self.nfe >= self.max_nfe

    @property
    def max_repetition_run_reached(self) -> bool:
        return self.max_repetition_run is not None and self.repetition_run_length >= self.max_repetition_run

    @property
    def is_completed(self) -> bool:
        return self.status == DllmReqStatus.COMPLETED

    def reset_new_tokens(self):
        self.new_tokens = 0

    def postprocess(self):
        pass

    def force_deactivate(self, reason=None):
        self.completion_reason = reason
        self.status = DllmReqStatus.COMPLETED


def test_scheduler_postprocess_kills_req_when_max_nfe_is_reached() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=7, max_nfe=2)
    scheduler.running_reqs.append(req)
    sample_output = SimpleNamespace(
        true_local_ids_map={"7": {}},
        accepted_ids_map={"7": {}},
        sampled_tokens_map={"7": {}},
        edit_writes_map={"7": {}},
    )

    scheduler.postprocess_multi_block([req], sample_output)
    assert req.nfe == 1
    assert req.status == DllmReqStatus.DECODING
    assert scheduler.freed_req_ids == []

    scheduler.postprocess_multi_block([req], sample_output)
    assert req.nfe == 2
    assert req.status == DllmReqStatus.FINISHED
    assert scheduler.freed_req_ids == [7]
    assert req not in scheduler.running_reqs


def test_scheduler_postprocess_kills_req_when_repetition_run_is_too_long() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=9, max_repetition_run=3)
    req.repetition_run_length = 3
    scheduler.running_reqs.append(req)
    sample_output = SimpleNamespace(
        true_local_ids_map={"9": {}},
        accepted_ids_map={"9": {}},
        sampled_tokens_map={"9": {}},
        edit_writes_map={"9": {}},
    )

    scheduler.postprocess_multi_block([req], sample_output)

    assert req.nfe == 1
    assert req.status == DllmReqStatus.FINISHED
    assert scheduler.freed_req_ids == [9]
    assert req not in scheduler.running_reqs


def test_repetition_run_length_counts_trailing_identical_tokens() -> None:
    req = SimpleNamespace(truncated_response=[11, 22, 22, 22])

    run_length = MultiBlockReqTemplate.repetition_run_length.fget(req)

    assert run_length == 3


def test_scheduler_postprocess_applies_edit_writes_map() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=11)
    req.token_ids = [9, 0, 0]

    class _Block:
        def __init__(self, req):
            self.req = req
            self.start = 0
            self.mask_token_id = 0

        @property
        def token_ids(self):
            return self.req.token_ids

        def write_token(self, token_id, rel_idx):
            self.req.token_ids[rel_idx] = token_id

    req.dllm_blocks = [_Block(req)]
    sample_output = SimpleNamespace(
        true_local_ids_map={"11": {}},
        accepted_ids_map={"11": {}},
        sampled_tokens_map={"11": {}},
        edit_writes_map={"11": {"0": {0: 0, 1: 5, 2: 6}}},
    )

    scheduler.postprocess_multi_block([req], sample_output)

    assert req.token_ids == [0, 5, 6]
    assert req.new_tokens == 2


def test_scheduler_postprocess_raises_when_req_id_map_is_missing() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=13)
    scheduler.running_reqs.append(req)
    sample_output = SimpleNamespace(
        true_local_ids_map={},
        accepted_ids_map={},
        sampled_tokens_map={},
        edit_writes_map={},
    )

    with pytest.raises(KeyError, match="13"):
        scheduler.postprocess_multi_block([req], sample_output)


def test_scheduler_postprocess_writes_block_trace_jsonl(tmp_path, monkeypatch) -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=17)
    req.token_ids = [9, 0, 0, 8]

    class _Block:
        def __init__(self, req, block_id, start, end, editable_start=0, status="ACTIVE"):
            self.req = req
            self.block_id = block_id
            self.start = start
            self.end = end
            self.editable_start = editable_start
            self.status = SimpleNamespace(name=status)
            self.mask_token_id = 0

        @property
        def token_ids(self):
            return self.req.token_ids[self.start : self.end]

        @property
        def num_mask_tokens(self):
            return sum(token_id == self.mask_token_id for token_id in self.token_ids)

        def write_token(self, token_id, rel_idx):
            self.req.token_ids[self.start + rel_idx] = token_id

    req.dllm_blocks = [_Block(req, 0, 0, 2), _Block(req, 1, 2, 4, editable_start=1)]

    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("DIFFULEX_BLOCK_TRACE_PATH", str(trace_path))
    monkeypatch.setenv("DIFFULEX_BLOCK_TRACE_MAX_NFE", "4")
    monkeypatch.setenv("DIFFULEX_BLOCK_TRACE_MAX_BLOCKS", "2")

    sample_output = SimpleNamespace(
        true_local_ids_map={"17": {}},
        accepted_ids_map={"17": {}},
        sampled_tokens_map={"17": {}},
        edit_writes_map={"17": {"0": {1: 5}}},
    )

    scheduler.postprocess_multi_block([req], sample_output)

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["req_id"] == 17
    assert record["step_nfe"] == 1
    assert record["new_tokens"] == 1
    assert record["trace_start_block_id"] == 1
    assert record["blocks"][0]["block_id"] == 1
    assert record["blocks"][0]["writes"] == {}
    assert record["blocks"][0]["token_ids"] == [0, 8]
    assert record["blocks"][0]["accepted_ids"] == []


def test_scheduler_postprocess_updates_block_commit_flags_from_sample_output() -> None:
    scheduler = _Scheduler()
    req = _Req(req_id=19)

    class _Block:
        def __init__(self):
            self.block_id = 0
            self.start = 0
            self.end = 2
            self.editable_start = 0
            self.status = SimpleNamespace(name="ACTIVE")
            self.mask_token_id = 0
            self.commit_ready = False
            self.same_as_previous = False
            self.all_confident = False

        @property
        def is_active(self):
            return True

        @property
        def num_mask_tokens(self):
            return 0

        @property
        def token_ids(self):
            return [1, 2]

        def write_token(self, token_id, rel_idx):
            raise AssertionError("write_token should not be called in this test")

    block = _Block()
    req.dllm_blocks = [block]
    sample_output = SimpleNamespace(
        true_local_ids_map={"19": {}},
        accepted_ids_map={"19": {}},
        sampled_tokens_map={"19": {}},
        edit_writes_map={"19": {}},
        block_state_map={"19": {"0": {"committable": True, "same_as_previous": False, "all_confident": True}}},
    )

    scheduler.postprocess_multi_block([req], sample_output)

    assert block.commit_ready is True
    assert block.same_as_previous is False
    assert block.all_confident is True


def test_multiblock_req_postprocess_requires_commit_ready_before_to_cache() -> None:
    class _Block:
        def __init__(self, commit_ready: bool):
            self.prev_block = None
            self.commit_ready = commit_ready
            self._is_to_cache = False

        @property
        def is_active(self):
            return not self._is_to_cache

        @property
        def is_complete(self):
            return True

        @property
        def is_to_cache(self):
            return self._is_to_cache

        @property
        def is_dummy(self):
            return False

        @property
        def is_in_cache(self):
            return False

        def to_cache(self):
            self._is_to_cache = True

    def _build_req(commit_ready: bool):
        block = _Block(commit_ready=commit_ready)
        req = SimpleNamespace(
            maybe_postprocess_prefix_blocks=lambda: None,
            dllm_block_buffer=SimpleNamespace(buffer_size=1, dllm_blocks=[block]),
            push_back_dummy_block=lambda: None,
            eos_token_generated=False,
            max_new_tokens_reached=False,
            max_model_len_reached=False,
            max_nfe_reached=False,
            max_repetition_run_reached=False,
        )
        return req, block

    req, block = _build_req(commit_ready=False)
    MultiBlockReqTemplate.postprocess(req)
    assert block.is_to_cache is False

    req, block = _build_req(commit_ready=True)
    MultiBlockReqTemplate.postprocess(req)
    assert block.is_to_cache is True
