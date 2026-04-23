from __future__ import annotations

import json
import os

from diffulex.engine.request import DllmReq
from diffulex.engine.scheduler import SchedulerBase
from diffulex.engine.status import DllmReqStatus
from diffulex.logger import get_logger
from diffulex.mixin.edit.scheduler import EditSchedulerMixin


class MultiBlockSchedulerTemplate(EditSchedulerMixin, SchedulerBase):
    _logger = get_logger(__name__)

    def init_multi_block(self: SchedulerBase) -> None:
        self.block_size = self.config.block_size

    def add_multi_block(self: SchedulerBase, req: DllmReq) -> None:
        req.init_multi_block(self.config)
        self.waiting_reqs.append(req)

    def schedule_multi_block(self: SchedulerBase) -> tuple[list[DllmReq], bool]:
        scheduled: list[DllmReq] = []
        num_reqs = 0
        num_batched_tokens = 0

        while self.waiting_reqs and num_reqs < self.max_num_reqs:
            req = self.waiting_reqs[0]

            projected = len(req) + self.block_size
            if num_batched_tokens + projected > self.max_num_batched_tokens or not self.kv_cache_manager.can_allocate(
                req
            ):
                break

            num_reqs += 1
            self.kv_cache_manager.allocate(req)
            req.apply_cached_prefix_pages()
            if req.is_preempted:
                if not self.kv_cache_manager.can_append(req):
                    self.kv_cache_manager.free(req)
                    num_reqs -= 1
                    break
                self.kv_cache_manager.may_append(req)

            num_batched_tokens += projected - req.num_cached_tokens
            req.make_pending()
            self.waiting_reqs.popleft()
            self.running_reqs.append(req)
            scheduled.append(req)

        if scheduled:
            return scheduled, True

        while self.running_reqs and num_reqs < self.max_num_reqs:
            req = self.running_reqs.popleft()
            while not self.kv_cache_manager.can_append(req):
                if self.running_reqs:
                    self.preempt(self.running_reqs.pop())
                else:
                    self.preempt(req)
                    break
            else:
                num_reqs += 1
                self.kv_cache_manager.may_append(req)
                scheduled.append(req)

        if not scheduled:
            diag = dict(
                phase="decode",
                waiting=len(self.waiting_reqs),
                running=len(self.running_reqs),
                max_num_reqs=self.max_num_reqs,
                max_num_batched_tokens=self.max_num_batched_tokens,
                block_size=self.block_size,
            )
            candidates = list(self.running_reqs)[:3] + list(self.waiting_reqs)[:2]
            details = []
            for idx, candidate in enumerate(candidates):
                try:
                    can_append = self.kv_cache_manager.can_append(candidate)
                except Exception:
                    can_append = "error"
                details.append(
                    f"[{idx}] status={candidate.status.name}, len={len(candidate)}, "
                    f"block_size={self.block_size}, "
                    f"new_tokens={candidate.new_tokens}, "
                    f"cached={candidate.num_cached_tokens}, "
                    f"can_append={can_append}"
                )
            raise RuntimeError(
                "MultiBlockScheduler: unable to schedule any req in decode; "
                f"state={diag}; details={' | '.join(details)}"
            )

        self.running_reqs.extendleft(reversed(scheduled))
        return scheduled, False

    def preempt_multi_block(self, req: DllmReq) -> None:
        req.preempt()
        self.kv_cache_manager.free(req)
        self.waiting_reqs.appendleft(req)

    @staticmethod
    def _trace_path() -> str | None:
        return os.getenv("DIFFULEX_BLOCK_TRACE_PATH")

    @staticmethod
    def _trace_max_nfe() -> int:
        raw = os.getenv("DIFFULEX_BLOCK_TRACE_MAX_NFE", "12")
        return max(0, int(raw))

    @staticmethod
    def _trace_max_blocks() -> int:
        raw = os.getenv("DIFFULEX_BLOCK_TRACE_MAX_BLOCKS", "3")
        return max(0, int(raw))

    def _append_block_trace(self, req: DllmReq, sample_output) -> None:
        trace_path = self._trace_path()
        if not trace_path:
            return

        max_nfe = self._trace_max_nfe()
        max_blocks = self._trace_max_blocks()
        if max_nfe <= 0 or max_blocks <= 0:
            return

        step_nfe = int(req.nfe) + 1
        if step_nfe > max_nfe:
            return

        req_id_str = str(req.req_id)
        edit_writes_map = sample_output.edit_writes_map.get(req_id_str, {})
        accepted_ids_map = sample_output.accepted_ids_map.get(req_id_str, {})
        dmax_trace_map = getattr(sample_output, "dmax_trace_map", {}).get(req_id_str, {})

        start_block_idx = 0
        for idx, block in enumerate(req.dllm_blocks):
            if getattr(block, "is_active", False):
                start_block_idx = idx
                break
            if int(getattr(block, "editable_start", 0) or 0) > 0:
                start_block_idx = idx
        traced_blocks = req.dllm_blocks[start_block_idx : start_block_idx + max_blocks]

        block_records = []
        for block in traced_blocks:
            rel_writes = edit_writes_map.get(str(block.block_id), {})
            block_records.append(
                {
                    "block_id": int(block.block_id),
                    "start": int(block.start),
                    "end": int(block.end),
                    "status": getattr(block.status, "name", str(block.status)),
                    "editable_start": int(getattr(block, "editable_start", 0) or 0),
                    "commit_ready": bool(getattr(block, "commit_ready", False)),
                    "same_as_previous": bool(getattr(block, "same_as_previous", False)),
                    "all_confident": bool(getattr(block, "all_confident", False)),
                    "num_mask_tokens": int(block.num_mask_tokens),
                    "token_ids": [int(token_id) for token_id in block.token_ids],
                    "writes": {int(rel_idx): int(token) for rel_idx, token in rel_writes.items()},
                    "accepted_ids": [int(idx) for idx in accepted_ids_map.get(str(block.block_id), [])],
                    "dmax_trace": dmax_trace_map.get(str(block.block_id)),
                }
            )

        record = {
            "req_id": int(req.req_id),
            "step_nfe": step_nfe,
            "req_status": getattr(req.status, "name", str(req.status)),
            "new_tokens": int(req.new_tokens),
            "completion_reason": req.completion_reason,
            "trace_start_block_id": int(traced_blocks[0].block_id) if traced_blocks else None,
            "blocks": block_records,
        }

        os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def postprocess_multi_block(
        self,
        reqs: list[DllmReq],
        sample_output,
    ) -> None:
        for req in reqs:
            req.reset_new_tokens()

            req_id_str = str(req.req_id)
            true_ids_map = sample_output.true_local_ids_map[req_id_str]
            accepted_ids_map = sample_output.accepted_ids_map[req_id_str]
            sampled_tokens_map = sample_output.sampled_tokens_map[req_id_str]
            for block_id, accepted_ids in accepted_ids_map.items():
                if not accepted_ids:
                    continue

                dllm_block = req.dllm_blocks[int(block_id)]
                sampled_tokens = sampled_tokens_map[block_id]
                true_local_ids = true_ids_map[block_id]
                for true_local_id, accepted_id in zip(true_local_ids, accepted_ids):
                    token = sampled_tokens[accepted_id]
                    dllm_block.write_token(token, true_local_id)
                req.new_tokens += len(accepted_ids)

            self.apply_edit_writes_map(req, sample_output)
            block_state_map = getattr(sample_output, "block_state_map", {}).get(req_id_str, {})
            for block in req.dllm_blocks:
                if not getattr(block, "is_active", False):
                    continue
                state = block_state_map.get(str(block.block_id), {})
                block.commit_ready = bool(state.get("committable", False))
                block.same_as_previous = bool(state.get("same_as_previous", False))
                block.all_confident = bool(state.get("all_confident", False))

            req.postprocess()
            self._append_block_trace(req, sample_output)
            req.nfe += 1
            if req.max_nfe_reached or req.max_repetition_run_reached:
                reason = "max_nfe_reached" if req.max_nfe_reached else "max_repetition_run_reached"
                req.force_deactivate(reason=reason)
            if req.is_completed:
                if req.completion_reason is None:
                    req.completion_reason = "completed_without_reason"
                self._logger.info(
                    "Req %s marked FINISHED (reason=%s, eos=%s, max_new=%s, max_model_len=%s, max_nfe=%s, max_repeat=%s, nfe=%s, gen_tokens=%s)",
                    req.req_id,
                    req.completion_reason,
                    req.eos_token_generated,
                    req.max_new_tokens_reached,
                    req.max_model_len_reached,
                    req.max_nfe_reached,
                    req.max_repetition_run_reached,
                    req.nfe,
                    len(req.truncated_response) if req.truncated_response is not None else -1,
                )
                req.status = DllmReqStatus.FINISHED
                self.kv_cache_manager.free(req)
                if req in self.running_reqs:
                    self.running_reqs.remove(req)
