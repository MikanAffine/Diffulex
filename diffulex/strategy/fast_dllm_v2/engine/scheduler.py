from __future__ import annotations

from diffulex.config import Config
from diffulex.engine.scheduler import AutoScheduler, SchedulerBase
from diffulex.engine.request import DllmReqStatus
from .request import FDV2Req


@AutoScheduler.register("fast_dllm_v2", is_default=True)
class FDV2Scheduler(SchedulerBase):
    def __init__(self, config: Config):
        super().__init__(config)
        self.block_size = config.diffusion_block_size

    def is_finished(self) -> bool:
        return not self.waiting_reqs and not self.running_reqs

    def add(self, req: FDV2Req) -> None:
        self.waiting_reqs.append(req)

    def schedule(self) -> tuple[list[FDV2Req], bool]:
        scheduled: list[FDV2Req] = []
        num_reqs = 0
        num_batched_tokens = 0
        while self.waiting_reqs and num_reqs < self.max_num_reqs:
            req = self.waiting_reqs[0]
            projected = len(req) + req.diffusion_block_size
            if (
                num_batched_tokens + projected > self.max_num_batched_tokens
                or not self.kv_cache_manager.can_allocate(req)
            ):
                break
            num_reqs += 1
            self.kv_cache_manager.allocate(req)
            num_batched_tokens += projected - req.num_cached_tokens
            req.status = DllmReqStatus.RUNNING
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
            diag = {
                "phase": "decode",
                "waiting": len(self.waiting_reqs),
                "running": len(self.running_reqs),
                "max_num_reqs": self.max_num_reqs,
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "block_size": self.block_size,
            }
            candidates = list(self.running_reqs)[:3] + list(self.waiting_reqs)[:2]
            details = []
            for idx, candidate in enumerate(candidates):
                try:
                    can_append = self.kv_cache_manager.can_append(candidate)
                except Exception:
                    can_append = "error"
                details.append(
                    f"[{idx}] status={candidate.status.name}, len={len(candidate)}, "
                    f"diff_block={getattr(candidate, 'diffusion_block_size', '?')}, "
                    f"new_tokens={getattr(candidate, 'new_tokens', '?')}, "
                    f"cached={getattr(candidate, 'num_cached_tokens', '?')}, "
                    f"can_append={can_append}"
                )
            raise RuntimeError(
                "FDV2Scheduler: unable to schedule any req in decode; "
                f"state={diag}; details={' | '.join(details)}"
            )
        self.running_reqs.extendleft(reversed(scheduled))
        return scheduled, False

    def preempt(self, req: FDV2Req) -> None:
        req.status = DllmReqStatus.WAITING
        self.kv_cache_manager.free(req)
        self.waiting_reqs.appendleft(req)

    def postprocess(
        self,
        reqs: list[FDV2Req],
        sample_output,
    ) -> dict[int, int]:
        n_diff_steps: dict[int, int] = {}
        for req in reqs:
            req.reset_new_tokens()
            req_id_str = str(req.req_id)
            true_ids_map = sample_output.true_local_ids_map.get(req_id_str, {})
            accepted_ids_map = sample_output.accepted_ids_map.get(req_id_str, {})
            sampled_tokens_map = sample_output.sampled_tokens_map.get(req_id_str, {})
            for block_id, accepted_ids in accepted_ids_map.items():
                if not accepted_ids:
                    continue
                diffusion_block = req.diffusion_blocks[int(block_id)]
                sampled_tokens = sampled_tokens_map.get(block_id, [])
                true_local_ids = true_ids_map.get(block_id, [])
                for true_local_id, accepted_id in zip(true_local_ids, accepted_ids):
                    token = sampled_tokens[accepted_id]
                    diffusion_block.modify_token(true_local_id, token)
                    if (
                        (not req.ignore_eos and token.item() == self.eos)
                        or req.num_completion_tokens >= req.max_tokens
                    ):
                        req.meet_eos = True
            if req.meet_eos and req.diffusion_blocks[-1].available_to_cache:
                req.status = DllmReqStatus.FINISHED
                self.kv_cache_manager.free(req)
                if req in self.running_reqs:
                    self.running_reqs.remove(req)
                n_diff_steps[req.req_id] = req.n_steps
            req.post_process()
        return n_diff_steps
