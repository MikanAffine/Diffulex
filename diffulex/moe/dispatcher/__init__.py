from diffulex.moe.dispatcher.datatype import CombineInput, DispatchOutput
from diffulex.moe.dispatcher.base import MoEDispatcher, TokenDispatcher
from diffulex.moe.dispatcher.trivial import TrivialTokenDispatcher

TrivialMoEDispatcher = TrivialTokenDispatcher


def build_dispatcher(
        impl: str,
        *args,
        **kwargs,
) -> TokenDispatcher:
    if impl == "trivial":
        return TrivialTokenDispatcher(*args, **kwargs)
    else:
        raise NotImplementedError


__all__ = [
    "CombineInput",
    "DispatchOutput",
    "TokenDispatcher",
    "TrivialTokenDispatcher",
    "build_dispatcher",
]
