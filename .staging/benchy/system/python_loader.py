"""`python:` — wrap an arbitrary Python callable/class as a System.

This is how a node, a workflow, or a LangGraph/DSPy agent enters benchy —
the vision's point that all four AI-system shapes are the same kind of
thing from outside. Accepts, in order of preference:

1. An object that already fully implements `benchy.core.System` (has
   `url`, `capabilities`, `invoke`, and `aclose`) — used unchanged, its own
   `url`/`capabilities` preserved.
2. A class (or instance) with an `invoke` method — wrapped so `aclose`
   falls back to a no-op if the target doesn't define one.
3. A plain callable `(request) -> Response | str | Awaitable[...]` — sync
   or async both work; a bare `str` return is wrapped into
   `Response(text=...)`.

URL grammar: `python:<path-or-dotted-module>[:<attr>]`.
`<path-or-dotted-module>` is a filesystem path (existing file, loaded via
`importlib.util.spec_from_file_location`) or a dotted module already
importable on `sys.path`. `<attr>` is looked up on the imported module; if
omitted, the first of `system`, `System`, `Agent` that exists wins, else
`load()` raises `LoadError` telling the caller to be explicit.

If `<attr>` names a class, it is instantiated with `**opts` (the kwargs
passed to `load()` beyond `capabilities`); a class that doesn't accept
those kwargs is retried with no arguments.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any

from benchy.core import Capabilities, LoadError, Request, Response
from benchy.core import System as CoreSystem
from benchy.system.base import BaseSystem

_DEFAULT_ATTR_NAMES = ("system", "System", "Agent")


def _import_target(target: str):
    path = Path(target)
    looks_like_path = target.endswith(".py") or "/" in target or path.exists()
    if looks_like_path:
        if not path.exists():
            raise LoadError(f"python: file not found: {target!r}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise LoadError(f"python: could not build an import spec for {target!r}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise LoadError(f"python: error importing {target!r}: {exc}") from exc
        return module

    try:
        return importlib.import_module(target)
    except Exception as exc:
        raise LoadError(f"python: could not import module {target!r}: {exc}") from exc


def _find_default_attr(module: Any, target: str) -> Any:
    for name in _DEFAULT_ATTR_NAMES:
        if hasattr(module, name):
            return getattr(module, name)
    raise LoadError(
        f"python: {target!r} has none of {_DEFAULT_ATTR_NAMES} — "
        f"name one explicitly, e.g. 'python:{target}:MyAgent'"
    )


class _PythonSystemAdapter(BaseSystem):
    """Wraps a class instance with `.invoke` or a plain callable as a System."""

    def __init__(self, target: Any, *, url: str, capabilities: Capabilities | None) -> None:
        super().__init__(url, capabilities or Capabilities())
        self._target = target

    async def invoke(self, request: Request) -> Response:
        invoke_fn = getattr(self._target, "invoke", None)
        result = invoke_fn(request) if invoke_fn is not None else self._target(request)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, str):
            return Response(text=result)
        if isinstance(result, Response):
            return result
        raise LoadError(
            f"python: {self._target!r} returned {type(result).__name__}, expected a Response or str"
        )

    async def aclose(self) -> None:
        close_fn = getattr(self._target, "aclose", None) or getattr(self._target, "close", None)
        if callable(close_fn):
            result = close_fn()
            if inspect.isawaitable(result):
                await result


def _instantiate(obj: Any, **opts: Any) -> Any:
    if not isinstance(obj, type):
        return obj
    try:
        return obj(**opts) if opts else obj()
    except TypeError:
        return obj()


def _wrap(obj: Any, *, url: str, capabilities: Capabilities | None, **opts: Any) -> CoreSystem:
    instance = _instantiate(obj, **opts)

    if isinstance(instance, CoreSystem):
        return instance  # already a full System — url/capabilities are its own

    if hasattr(instance, "invoke"):
        caps = getattr(instance, "capabilities", None) or capabilities
        return _PythonSystemAdapter(instance, url=getattr(instance, "url", None) or url, capabilities=caps)

    if callable(instance):
        return _PythonSystemAdapter(instance, url=url, capabilities=capabilities)

    raise LoadError(f"python: {instance!r} is not callable and has no invoke method")


def load_python(rest: str, **opts: Any) -> CoreSystem:
    if not rest:
        raise LoadError(
            "python: URL must include a file path or dotted module, "
            "e.g. 'python:./my_agent.py:Agent'"
        )
    target, sep, attr = rest.rpartition(":")
    if not sep:
        target, attr = rest, ""

    module = _import_target(target)
    if attr:
        if not hasattr(module, attr):
            raise LoadError(f"python: {target!r} has no attribute {attr!r}")
        obj = getattr(module, attr)
    else:
        obj = _find_default_attr(module, target)

    capabilities = opts.pop("capabilities", None)
    return _wrap(obj, url=f"python:{rest}", capabilities=capabilities, **opts)
