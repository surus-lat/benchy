"""The scorer registry: `register_scorer`, `registry`, and `parse_scorer`.

Every scorer's `__repr__` is, by construction, a Python call expression that
names a registered factory and passes it literal (or nested-scorer) keyword
arguments — e.g. `binary(field_wise(fields=('vendor', 'total'),
per_field=exact_match()))`. `parse_scorer` evaluates exactly that grammar
back into a live scorer: a restricted `eval` whose only names are the
registry's factories and whose only builtins are none at all. That
restriction is deliberate — `parse_scorer` is not a general-purpose
deserializer for untrusted strings, it is the round-trip half of "reading a
rubric's repr tells you what it means, and that repr reconstructs it."
Benchmark YAML files that embed a scorer repr are as trusted as the rest of
the benchmark definition.

`register_scorer` is also how a contributor adds a new *family* (the rare
activity the vision calls out) to the grammar `parse_scorer` accepts:

    @register_scorer("my_family")
    def my_family(...) -> Scorer: ...

after which `my_family(...)` reprs round-trip like every built-in primitive.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from benchy.core import Scorer

__all__ = ["registry", "register_scorer", "parse_scorer"]

#: name -> factory. Mutated only through `register_scorer`.
registry: dict[str, Callable[..., Scorer]] = {}


def register_scorer(
    name: str, factory: Callable[..., Scorer] | None = None
) -> Callable[..., Scorer]:
    """Register `factory` under `name` in the global scorer registry.

    Two call shapes:

        register_scorer("my_family", my_family)     # direct

        @register_scorer("my_family")                # decorator
        def my_family(...): ...

    Re-registering the exact same factory under a name it already owns is a
    no-op (idempotent module reloads). Registering a *different* factory
    under a name that is already taken raises `ValueError` — silently
    shadowing a name would make `parse_scorer` non-deterministic depending
    on import order.
    """

    def _register(fn: Callable[..., Scorer]) -> Callable[..., Scorer]:
        existing = registry.get(name)
        if existing is not None and existing is not fn:
            raise ValueError(
                f"scorer name {name!r} is already registered to a different factory "
                f"({existing!r}); pick a different name"
            )
        registry[name] = fn
        return fn

    if factory is None:
        return _register
    return _register(factory)


def parse_scorer(text: str) -> Scorer:
    """Reconstruct a scorer from its `repr`.

    `parse_scorer(repr(scorer)) == scorer` for every scorer this package
    builds. Raises `ValueError` if `text` is not a well-formed expression
    over the registry, or if it evaluates to something that isn't a
    `Scorer`.
    """
    try:
        code = compile(text, "<scorer-repr>", "eval")
        result: Any = eval(code, {"__builtins__": {}}, dict(registry))  # noqa: S307
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not parse scorer expression {text!r}: {exc}") from exc

    if not isinstance(result, Scorer):
        raise ValueError(
            f"expression {text!r} evaluated to {type(result).__name__!r}, not a Scorer"
        )
    return result
