"""BaseSystem — shared boilerplate for System implementations.

`benchy.core.System` is a structural (`runtime_checkable`) Protocol, so
nothing requires subclassing this. Every scheme shipped in this package
does anyway, because storing `url`/`capabilities` and providing a safe
default `aclose` is the same handful of lines in each of them.
"""

from __future__ import annotations

from benchy.core import Capabilities, Request, Response


class BaseSystem:
    """Minimal concrete System: stores `url`/`capabilities`, no-op `aclose`.

    Subclasses override `invoke`. Systems with real resources to release
    (an HTTP client, a loaded model, a subprocess) override `aclose` too;
    systems with nothing to release (in-process, stateless, like `echo:`)
    can leave it as the inherited no-op.
    """

    def __init__(self, url: str, capabilities: Capabilities | None = None) -> None:
        self.url = url
        self.capabilities = capabilities if capabilities is not None else Capabilities()

    async def invoke(self, request: Request) -> Response:  # pragma: no cover - abstract
        raise NotImplementedError

    async def aclose(self) -> None:
        return None
