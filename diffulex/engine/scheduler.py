from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque, defaultdict
from typing import Callable

from diffulex.config import Config
from diffulex.engine.kv_cache_manager import AutoKVCacheManager
from diffulex.engine.request import DllmReq
from diffulex.engine.status import DllmReqStatus
from diffulex.engine.strategy_registry import DiffulexStrategyRegistry


class SchedulerBase(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.max_num_reqs = config.max_num_reqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.kv_cache_manager = AutoKVCacheManager.from_config(config)
        self.waiting_reqs: deque[DllmReq] = deque()
        self.running_reqs: deque[DllmReq] = deque()

    def is_finished(self) -> bool:
        return not self.waiting_reqs and not self.running_reqs

    def abort_request(self, req_id: int) -> bool:
        for req in list(self.waiting_reqs):
            if req.req_id == req_id:
                self.waiting_reqs.remove(req)
                req.status = DllmReqStatus.FINISHED
                setattr(req, "completion_reason", "aborted")
                return True

        for req in list(self.running_reqs):
            if req.req_id == req_id:
                self.running_reqs.remove(req)
                setattr(req, "completion_reason", "aborted")
                req.status = DllmReqStatus.FINISHED
                self.kv_cache_manager.free(req)
                return True

        return False

    @abstractmethod
    def add(self, req: DllmReq) -> None:
        pass

    @abstractmethod
    def schedule(self) -> tuple[list[DllmReq], bool]:
        pass

    @abstractmethod
    def preempt(self, req: DllmReq) -> None:
        pass

    @abstractmethod
    def postprocess(self, reqs: list[DllmReq], sampler_output):
        pass


class DataParallelScheduler:
    def __init__(self, config: Config, scheduler_factory: Callable[[Config], SchedulerBase]):
        self.config = config
        self.dp_size = config.data_parallel_size
        self.schedulers = [scheduler_factory(config) for _ in range(self.dp_size)]

    def _owner_for_req(self, req: DllmReq) -> int:
        owner_assigned = bool(getattr(req, "_dp_owner_assigned", False))
        owner = getattr(req, "dp_rank", None)
        if not owner_assigned or owner is None or not (0 <= owner < self.dp_size):
            owner = req.req_id % self.dp_size
            assign_fn = getattr(req, "assign_dp_rank", None)
            if callable(assign_fn):
                assign_fn(owner)
            else:
                req.dp_rank = owner
        return owner

    def add(self, req: DllmReq) -> None:
        owner = self._owner_for_req(req)
        self.schedulers[owner].add(req)

    def schedule(self) -> tuple[list[DllmReq], bool]:
        scheduled: list[DllmReq] = []
        saw_prefill = False
        for scheduler in self.schedulers:
            if scheduler.is_finished():
                continue
            local_reqs, is_prefill = scheduler.schedule()
            scheduled.extend(local_reqs)
            saw_prefill = saw_prefill or is_prefill
        return scheduled, saw_prefill

    def postprocess(self, reqs: list[DllmReq], sampler_output) -> None:
        reqs_by_owner: dict[int, list[DllmReq]] = defaultdict(list)
        for req in reqs:
            reqs_by_owner[self._owner_for_req(req)].append(req)

        for owner, local_reqs in reqs_by_owner.items():
            if local_reqs:
                self.schedulers[owner].postprocess(local_reqs, sampler_output)

    def is_finished(self) -> bool:
        return all(scheduler.is_finished() for scheduler in self.schedulers)

    def abort_request(self, req_id: int) -> bool:
        for scheduler in self.schedulers:
            if scheduler.abort_request(req_id):
                return True
        return False


SchedulerFactory = Callable[[Config], SchedulerBase]


class AutoScheduler(DiffulexStrategyRegistry):
    """Registry-driven factory for scheduler implementations."""

    @classmethod
    def _from_single_config(cls, config: Config) -> SchedulerBase:
        cls._ensure_strategies_loaded()
        cls._MODULE_MAPPING: dict[str, SchedulerFactory]
        candidates: list[str] = []
        if config.decoding_strategy:
            candidates.append(config.decoding_strategy)
        candidates.append(cls._DEFAULT_KEY)

        for key in candidates:
            factory = cls._MODULE_MAPPING.get(key)
            if factory is not None:
                return factory(config)

        available = ", ".join(cls.available_modules()) or "<none>"
        raise ValueError(
            "No scheduler registered for decoding_strategy="
            f"'{config.decoding_strategy}'. Available schedulers: {available}."
        )

    @classmethod
    def from_config(cls, config: Config) -> SchedulerBase | DataParallelScheduler:
        if config.data_parallel_size > 1:
            return DataParallelScheduler(config, scheduler_factory=cls._from_single_config)
        return cls._from_single_config(config)
