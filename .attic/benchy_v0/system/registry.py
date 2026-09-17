"""The pluggable URL-scheme registry: `scheme:rest` -> `System`.

`load()` is the one function every other benchy module calls to turn a
config string into a live `benchy.core.System`. It never imports a heavy
dependency itself — that's each loader's job, and each loader is expected
to import lazily so `import benchy.system` stays instant and dependency-free.
"""

from __future__ import annotations

from benchy.core import LoadError, SystemLoader

_REGISTRY: dict[str, SystemLoader] = {}


def register(scheme: str, loader: SystemLoader) -> None:
    """Register (or replace) the loader for a URL scheme.

    Re-registering an already-known scheme silently replaces it — this is
    the pluggability the vision asks for: a user can swap out `openai:`'s
    loader, or add a brand new scheme, without touching this module.
    """
    if not scheme or ":" in scheme:
        raise LoadError(f"invalid scheme name {scheme!r}: must be non-empty and contain no ':'")
    _REGISTRY[scheme] = loader


def schemes() -> list[str]:
    """The currently registered scheme names, sorted."""
    return sorted(_REGISTRY)


def load(url: str, **opts: object) -> "System":  # noqa: F821 - see below
    """Parse `scheme:rest`, dispatch to the registered loader, return a System.

    Raises `benchy.core.LoadError` if `url` has no scheme, or if the scheme
    is not registered. The error message always lists the known schemes —
    this is a developer-facing tool and a confused developer needs the menu,
    not just "no".
    """
    if not isinstance(url, str) or ":" not in url:
        raise LoadError(
            f"malformed system URL {url!r}: expected 'scheme:rest' "
            f"(known schemes: {', '.join(schemes()) or '<none registered>'})"
        )
    scheme, _, rest = url.partition(":")
    if not scheme:
        raise LoadError(
            f"malformed system URL {url!r}: missing scheme before ':' "
            f"(known schemes: {', '.join(schemes()) or '<none registered>'})"
        )
    loader = _REGISTRY.get(scheme)
    if loader is None:
        raise LoadError(
            f"unknown system scheme {scheme!r} in {url!r} "
            f"(known schemes: {', '.join(schemes()) or '<none registered>'})"
        )
    return loader(rest, **opts)
