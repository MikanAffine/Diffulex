import inspect
import functools

from typing import Any


def fetch_factory_name(factory: Any) -> str:
    # unwrap decorated functions (inspect.unwrap works for functions with __wrapped__)
    try:
        orig = inspect.unwrap(factory)
    except Exception:
        orig = factory

    # handle functools.partial
    if isinstance(orig, functools.partial):
        return fetch_factory_name(orig.func)

    # class
    if inspect.isclass(orig):
        name = getattr(orig, "__qualname__", orig.__name__)
        module = getattr(orig, "__module__", "")
    # function
    elif inspect.isfunction(orig):
        name = getattr(orig, "__qualname__", orig.__name__)
        module = getattr(orig, "__module__", "")
    else:
        # callable instance (object with __call__)
        name = getattr(orig, "__name__", None) or orig.__class__.__name__
        module = getattr(orig, "__module__", getattr(orig.__class__, "__module__", ""))

    if module and module != "builtins":
        return f"{module}.{name}"
    return name
