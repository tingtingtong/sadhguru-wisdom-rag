import functools
from typing import Any, Callable

import logfire


def trace(name: str | None = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__qualname__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with logfire.span(span_name):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with logfire.span(span_name):
                return func(*args, **kwargs)

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
