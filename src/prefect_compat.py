"""Compatibility wrapper for optional Prefect dependency."""

from __future__ import annotations

import os
from typing import Any, Callable

PREFECT_AVAILABLE = False
_DISABLE_PREFECT = os.environ.get("BENCHY_DISABLE_PREFECT", "").lower() in (
    "1",
    "true",
    "yes",
)

if not _DISABLE_PREFECT:
    try:
        from prefect import flow as _prefect_flow
        from prefect import serve as _prefect_serve
        from prefect import task as _prefect_task
        from prefect.cache_policies import NO_CACHE as _PREFECT_NO_CACHE

        PREFECT_AVAILABLE = True
        flow = _prefect_flow
        task = _prefect_task
        serve = _prefect_serve
        NO_CACHE = _PREFECT_NO_CACHE
    except Exception:
        _DISABLE_PREFECT = True

if _DISABLE_PREFECT:
    NO_CACHE = None

    def _identity_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    def flow(*args: Any, **kwargs: Any) -> Callable[..., Any]:
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def task(*args: Any, **kwargs: Any) -> Callable[..., Any]:
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def serve(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "Prefect is disabled or not installed. Install with `pip install .[prefect]` "
            "and unset BENCHY_DISABLE_PREFECT to enable flow registration."
        )
