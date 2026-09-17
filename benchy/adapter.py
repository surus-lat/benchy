"""The runtime boundary — the one interface the engine knows (paper A.10).

    named-field input object  ->  named-field output object

That is the whole contract. Everything an integration needs to do — credentials,
HTTP, SDK construction, prompt assembly, reading an artifact off disk, pulling the
answer out of a provider response — happens on the far side of this line, which is
why the engine has no provider branches and no `type: model` execution path.

The engine owns validation, scoring, aggregation and diagnostics. The adapter owns
translation. `invoke` is the only method used per example.

**The registry is deliberately off the execution path.** A run evaluates exactly one
AI-system, so `run()` takes its adapter directly; there is no module-global state
between the engine and the thing under test. `bind`/`resolve` exist for callers —
the CLI, a server — that must turn an AI-system *definition* into an implementation,
which is the runtime's business rather than the engine's (A.11).
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

from benchy.errors import BenchyError

__all__ = ["Adapter", "invoker", "bind", "resolve", "clear", "key_for"]


class Adapter(Protocol):
    """What an integration implements. A plain callable is accepted too."""

    def invoke(self, input_object: dict[str, object]) -> Awaitable[dict[str, object]]:
        ...


def invoker(adapter: object) -> Callable[[dict], Awaitable[object]]:
    """Normalize anything adapter-shaped into one awaitable call.

    Accepts an object with `invoke`, or a bare callable; sync or async either way.
    Six lines here is the whole reason no adapter base class, wrapper class or
    `FunctionAdapter` needs to exist.
    """
    call = getattr(adapter, "invoke", adapter)
    if not callable(call):
        raise BenchyError(
            "runtime", "adapter_not_bound",
            f"adapter must be callable or expose invoke(); got {type(adapter).__name__}",
        )

    async def invoke(input_object: dict) -> object:
        result = call(input_object)
        return await result if inspect.isawaitable(result) else result

    return invoke


# ---------------------------------------------------------------------------
# binding: AI-system definition -> implementation (A.11)
# ---------------------------------------------------------------------------

_BINDINGS: dict[str, object] = {}


def key_for(system: Mapping) -> str:
    """The binding key for an AI-system definition.

    `type` prefixes the name so an external id can never collide with a provider.
    """
    return f"external:{system['id']}" if system.get("type") == "external" else f"model:{system.get('provider')}"


def bind(key: str, adapter: object) -> None:
    """Register an implementation for an AI-system, by `key_for` key."""
    _BINDINGS[key] = adapter


def resolve(system: Mapping) -> object:
    """The implementation bound for this AI-system, or a run setup error."""
    key = key_for(system)
    if key not in _BINDINGS:
        raise BenchyError(
            "runtime", "adapter_not_bound",
            f"no adapter bound for {key!r}; bound: {', '.join(sorted(_BINDINGS)) or '<none>'}",
            ["ai-system"],
        )
    return _BINDINGS[key]


def clear() -> None:
    _BINDINGS.clear()
