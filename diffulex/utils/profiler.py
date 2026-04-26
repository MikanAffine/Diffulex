from __future__ import annotations

import functools
import inspect
import logging
import os
import threading
import contextvars
from dataclasses import asdict, dataclass
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Callable, ParamSpec, TypeVar

import torch

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; falling back to %s.", name, value, default)
        return default


# Read once at import time so the decorator path is static and branch-free when off.
PROFILE_ENABLED = _env_flag("DIFFULEX_PROFILE", False)
_TRACE_PHASE: contextvars.ContextVar[str | None] = contextvars.ContextVar("diffulex_trace_phase", default=None)


def trace_phase_from_prefill_flags(is_prefill: Any) -> str | None:
    if isinstance(is_prefill, bool):
        return "prefill" if is_prefill else "decode"

    if torch.is_tensor(is_prefill):
        if is_prefill.numel() == 0:
            return None
        flags = is_prefill.to(dtype=torch.bool)
        all_prefill = bool(flags.all().item())
        any_prefill = bool(flags.any().item())
    else:
        try:
            flags = [bool(flag) for flag in is_prefill]
        except TypeError:
            return None
        if not flags:
            return None
        all_prefill = all(flags)
        any_prefill = any(flags)

    if all_prefill:
        return "prefill"
    if not any_prefill:
        return "decode"
    return "mixed"


def trace_phase_from_attn_metadata(attn_metadata: Any) -> str | None:
    phase = trace_phase_from_prefill_flags(getattr(attn_metadata, "is_prefill", None))
    if phase is not None:
        return phase

    status_table = getattr(attn_metadata, "status_table", None)
    if status_table is None:
        return None

    if torch.is_tensor(status_table):
        return trace_phase_from_prefill_flags(status_table == 0)

    try:
        return trace_phase_from_prefill_flags([int(status) == 0 for status in status_table])
    except TypeError:
        return trace_phase_from_prefill_flags(int(status_table) == 0)


def _callable_display_name(func: Callable[..., Any]) -> str:
    try:
        target = inspect.unwrap(func)
    except Exception:
        target = func
    return getattr(target, "__qualname__", getattr(target, "__name__", "anonymous"))


def current_trace_phase() -> str | None:
    return _TRACE_PHASE.get()


@contextmanager
def trace_phase_scope(phase: str | None):
    token = _TRACE_PHASE.set(phase)
    try:
        yield
    finally:
        _TRACE_PHASE.reset(token)


@dataclass(slots=True)
class ProfilerSettings:
    output_dir: str = "./log/profiler"
    wait: int = 1
    warmup: int = 1
    active: int = 3
    repeat: int = 1
    record_shapes: bool = True
    with_stack: bool = False
    with_flops: bool = False
    profile_memory: bool = False
    use_gzip: bool = True

    @classmethod
    def from_env(cls) -> "ProfilerSettings":
        return cls(
            output_dir=os.getenv("DIFFULEX_PROFILE_DIR", "./log/profiler"),
            wait=_env_int("DIFFULEX_PROFILE_WAIT", 1),
            warmup=_env_int("DIFFULEX_PROFILE_WARMUP", 1),
            active=_env_int("DIFFULEX_PROFILE_ACTIVE", 3),
            repeat=_env_int("DIFFULEX_PROFILE_REPEAT", 1),
            record_shapes=_env_flag("DIFFULEX_PROFILE_RECORD_SHAPES", True),
            with_stack=_env_flag("DIFFULEX_PROFILE_WITH_STACK", False),
            with_flops=_env_flag("DIFFULEX_PROFILE_WITH_FLOPS", False),
            profile_memory=_env_flag("DIFFULEX_PROFILE_MEMORY", False),
            use_gzip=_env_flag("DIFFULEX_PROFILE_GZIP", True),
        )


def profiler_enabled() -> bool:
    return PROFILE_ENABLED


class ProfilerManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._profiler: torch.profiler.profile | None = None
        self._settings = ProfilerSettings()
        self._worker_name: str | None = None
        self._output_path: Path | None = None

    @property
    def is_active(self) -> bool:
        return self._profiler is not None

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    def status(self) -> dict[str, Any]:
        return {
            "active": self.is_active,
            "output_path": str(self._output_path) if self._output_path is not None else None,
            "worker_name": self._worker_name,
            "settings": asdict(self._settings),
        }

    def _build_activities(self) -> list[torch.profiler.ProfilerActivity]:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        return activities

    def _resolve_settings(
        self,
        *,
        output_dir: str | None = None,
        wait: int | None = None,
        warmup: int | None = None,
        active: int | None = None,
        repeat: int | None = None,
        record_shapes: bool | None = None,
        with_stack: bool | None = None,
        with_flops: bool | None = None,
        profile_memory: bool | None = None,
        use_gzip: bool | None = None,
    ) -> ProfilerSettings:
        env = ProfilerSettings.from_env()
        return ProfilerSettings(
            output_dir=output_dir or env.output_dir,
            wait=env.wait if wait is None else wait,
            warmup=env.warmup if warmup is None else warmup,
            active=env.active if active is None else active,
            repeat=env.repeat if repeat is None else repeat,
            record_shapes=env.record_shapes if record_shapes is None else record_shapes,
            with_stack=env.with_stack if with_stack is None else with_stack,
            with_flops=env.with_flops if with_flops is None else with_flops,
            profile_memory=env.profile_memory if profile_memory is None else profile_memory,
            use_gzip=env.use_gzip if use_gzip is None else use_gzip,
        )

    def start(
        self,
        *,
        worker_name: str | None = None,
        output_dir: str | None = None,
        wait: int | None = None,
        warmup: int | None = None,
        active: int | None = None,
        repeat: int | None = None,
        record_shapes: bool | None = None,
        with_stack: bool | None = None,
        with_flops: bool | None = None,
        profile_memory: bool | None = None,
        use_gzip: bool | None = None,
    ) -> bool:
        with self._lock:
            if self.is_active:
                self.stop()

            self._settings = self._resolve_settings(
                output_dir=output_dir,
                wait=wait,
                warmup=warmup,
                active=active,
                repeat=repeat,
                record_shapes=record_shapes,
                with_stack=with_stack,
                with_flops=with_flops,
                profile_memory=profile_memory,
                use_gzip=use_gzip,
            )
            self._worker_name = worker_name

            trace_dir = Path(self._settings.output_dir)
            if worker_name:
                trace_dir = trace_dir / worker_name
            trace_dir.mkdir(parents=True, exist_ok=True)
            self._output_path = trace_dir

            schedule = torch.profiler.schedule(
                wait=max(0, int(self._settings.wait)),
                warmup=max(0, int(self._settings.warmup)),
                active=max(1, int(self._settings.active)),
                repeat=max(1, int(self._settings.repeat)),
            )
            profiler = torch.profiler.profile(
                activities=self._build_activities(),
                schedule=schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(trace_dir),
                    worker_name=worker_name,
                    use_gzip=bool(self._settings.use_gzip),
                ),
                record_shapes=bool(self._settings.record_shapes),
                with_stack=bool(self._settings.with_stack),
                with_flops=bool(self._settings.with_flops),
                profile_memory=bool(self._settings.profile_memory),
            )
            try:
                profiler.start()
            except Exception:
                logger.exception("Failed to start torch.profiler; profiling remains disabled.")
                self._profiler = None
                self._output_path = None
                return False

            self._profiler = profiler
            logger.info(
                "Profiler started for %s -> %s (gzip=%s)",
                worker_name or "default",
                trace_dir,
                self._settings.use_gzip,
            )
            return True

    def start_if_enabled(self, *, worker_name: str | None = None, use_gzip: bool | None = None) -> bool:
        if not profiler_enabled():
            return False
        if self.is_active:
            return True
        return self.start(worker_name=worker_name, use_gzip=use_gzip)

    def step(self) -> None:
        profiler = self._profiler
        if profiler is None:
            return
        try:
            profiler.step()
        except Exception:
            logger.debug("Profiler step failed.", exc_info=True)

    def stop(self) -> None:
        with self._lock:
            profiler = self._profiler
            if profiler is None:
                return
            self._profiler = None
            try:
                profiler.stop()
            except Exception:
                logger.debug("Profiler stop failed.", exc_info=True)
            finally:
                logger.info("Profiler stopped for %s", self._worker_name or "default")
                self._worker_name = None


_PROFILER = ProfilerManager()


def get_profiler() -> ProfilerManager:
    return _PROFILER


def _resolve_trace_name(func: Callable[..., Any], alias: str | None = None) -> str:
    base_name = alias or _callable_display_name(func)
    phase = current_trace_phase()
    if phase and not base_name.startswith(("prefill/", "decode/", "mixed/")):
        return f"{phase}/{base_name}"
    return base_name


def _decorate_callable(func: Callable[P, R] | Callable[P, Any], alias: str | None = None):
    if not PROFILE_ENABLED:
        return func

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs):
            profiler = get_profiler()
            if not profiler.is_active:
                return await func(*args, **kwargs)
            name = _resolve_trace_name(func, alias)
            with torch.profiler.record_function(name):
                return await func(*args, **kwargs)

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        profiler = get_profiler()
        if not profiler.is_active:
            return func(*args, **kwargs)
        name = _resolve_trace_name(func, alias)
        with torch.profiler.record_function(name):
            return func(*args, **kwargs)

    return wrapper


def trace(name: str | Callable[..., Any] | None = None):
    if callable(name) and not isinstance(name, str):
        return _decorate_callable(name)

    if not PROFILE_ENABLED:
        def identity_decorator(func):
            return func

        return identity_decorator

    def decorator(func):
        return _decorate_callable(func, alias=name)

    return decorator


__all__ = [
    "ProfilerManager",
    "ProfilerSettings",
    "current_trace_phase",
    "get_profiler",
    "profiler_enabled",
    "trace_phase_from_attn_metadata",
    "trace_phase_from_prefill_flags",
    "trace_phase_scope",
    "trace",
]
