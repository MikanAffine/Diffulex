from diffulex.moe.runner.base import MoERunner
from diffulex.moe.runner.trivial import TrivialMoERunner


def build_runner(
        impl: str,
        *args,
        **kwargs,
) -> MoERunner:
    if impl == "trivial":
        return TrivialMoERunner(*args, **kwargs)
    else:
        raise NotImplementedError


__all__ = [
    "MoERunner",
    "TrivialMoERunner",
    "build_runner",
]
