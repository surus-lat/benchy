"""`echo:` / `mock:` — the dependency-free test double.

This is the first scheme built (per the build brief, siblings are blocked
on it) and its constructor kwargs are a **frozen cross-module contract**:
`tests/benchy/integration/test_seams.py` (the round-2 merge gate) drives
`EchoSystem` with these exact keyword arguments, verbatim:

    load("echo:")                                                  # sane defaults
    load("echo:", text="42")                                       # every invoke -> Response(text="42")
    load("echo:", data={"name": "Ana"})                            # every invoke -> Response(data={...})
    load("echo:", capabilities=Capabilities(structured_output=True))
    load("echo:", text="...", capabilities=Capabilities(structured_output=False))
    load("echo:", responses=[{"name": "Ana"}, {"name": "Beto"}, {"name": "Carla"}])
    load("echo:", error_on=["2"], data={"name": "Ana"}, capabilities=Capabilities(...))

Semantics:

- `text=` sets `Response.text`; `data=` sets `Response.data`. Both may be
  given together.
- `capabilities=` replaces the advertised `Capabilities` wholesale — this
  is how a sibling test drives capability negotiation.
- `responses=` is a scripted sequence, one item consumed per `invoke`,
  **cycling** back to the start once exhausted (a scripted run may be
  re-run: `Benchmark.compare()` grades several systems against the same
  script, `as_loss()` re-runs the same benchmark on every optimizer step).
  Each item may be a `Response` (used as-is), a `str` (-> `Response(text=..)`),
  a `dict` (-> `Response(data=..)`), or an exception / exception type
  (raised instead of returned).
- `error_on=` is a collection of sample ids. `EchoSystem` does not see a
  `Sample`, only a `Request` — so a match is looked up in
  `request.meta["sample_id"]` first, falling back to scanning
  `request.meta.values()` if that key is absent. A match returns a
  `Response` with `.error` set; it is never raised — see the module-wide
  error-handling policy below.
- `raises=` (an exception instance or type) makes every `invoke` raise
  instead of returning — for exercising a caller's exception handling.
  This is the one way to make an `echo:` system misbehave like a truly
  broken transport; everything else funnels into a returned `Response`.
- `latency_ms=` sleeps that long before responding and reports the
  (measured) elapsed time on `Response.latency_ms`.
- `responder=` is an escape hatch: a `(Request) -> Response | str | dict`
  callable (sync or async) computed fresh per call.
- Every `Request` received is recorded, in order, on `system.requests`
  (`system.received` is a read-only alias of the same list).

**Error-handling policy** (see `benchy/system/__init__.py` for the
module-wide statement): a System's `invoke` returns a `Response` with
`.error` set for anything that is "the request failed, gracefully" —
never raises for that. `raises=`/exception items in `responses=` exist
here specifically to let a caller's exception-handling *path* be tested;
real schemes (`openai:`, `endpoint:`, `hf:`) never raise from `invoke`
for a provider-side failure either.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any

from benchy.core import Capabilities, Request, Response, TextPart
from benchy.system.base import BaseSystem

ResponseLike = Response | str | dict[str, Any] | BaseException | type[BaseException]
Responder = Callable[[Request], "Response | str | dict[str, Any] | Awaitable[Any]"]


def _coerce(item: ResponseLike) -> Response:
    """Turn one scripted `responses=` item (or a `responder=` result) into a Response."""
    if isinstance(item, Response):
        return item
    if isinstance(item, type) and issubclass(item, BaseException):
        raise item()
    if isinstance(item, BaseException):
        raise item
    if isinstance(item, str):
        return Response(text=item)
    if isinstance(item, dict):
        return Response(data=item)
    # Anything else (list, number, dataclass, ...) is still "structured data".
    return Response(data=item)


class EchoSystem(BaseSystem):
    """Deterministic, in-process System double. See module docstring."""

    def __init__(
        self,
        url: str = "echo:",
        *,
        text: str | None = None,
        data: Any = None,
        responses: Sequence[ResponseLike] | None = None,
        error: str | None = None,
        error_on: Iterable[Any] | None = None,
        raises: BaseException | type[BaseException] | None = None,
        latency_ms: float = 0.0,
        responder: Responder | None = None,
        capabilities: Capabilities | None = None,
    ) -> None:
        super().__init__(url, capabilities)
        self._text = text
        self._data = data
        self._responses = list(responses) if responses is not None else None
        self._error = error
        self._error_on = {str(x) for x in error_on} if error_on else None
        self._raises = raises
        self._latency_ms = float(latency_ms)
        self._responder = responder
        self._index = 0
        self.requests: list[Request] = []
        self.closed = False

    @property
    def received(self) -> list[Request]:
        """Read-only alias of `requests`, for call-site readability."""
        return self.requests

    def _matched_error_id(self, request: Request) -> str | None:
        if not self._error_on:
            return None
        sample_id = request.meta.get("sample_id")
        if sample_id is not None:
            sid = str(sample_id)
            return sid if sid in self._error_on else None
        for value in request.meta.values():
            sval = str(value)
            if sval in self._error_on:
                return sval
        return None

    async def invoke(self, request: Request) -> Response:
        self.requests.append(request)
        start = time.perf_counter()

        if self._latency_ms:
            await asyncio.sleep(self._latency_ms / 1000)

        def finish(response: Response) -> Response:
            if response.latency_ms is not None:
                return response
            return dataclasses.replace(response, latency_ms=(time.perf_counter() - start) * 1000)

        if self._raises is not None:
            exc = self._raises
            raise exc() if isinstance(exc, type) else exc

        matched_id = self._matched_error_id(request)
        if matched_id is not None:
            message = self._error or f"echo: induced error for sample {matched_id!r}"
            return finish(Response(error=message))

        if self._responder is not None:
            result = self._responder(request)
            if inspect.isawaitable(result):
                result = await result
            return finish(_coerce(result))

        if self._responses is not None:
            if not self._responses:
                raise ValueError("EchoSystem: responses=[] is empty; provide at least one scripted response")
            item = self._responses[self._index % len(self._responses)]
            self._index += 1
            return finish(_coerce(item))

        if self._error is not None:
            return finish(Response(error=self._error))

        if self._text is not None or self._data is not None:
            return finish(Response(text=self._text, data=self._data))

        # Zero-config default: actually echo the request's text back. Audio/
        # image-only requests echo as "" rather than raising — a System never
        # sees or enforces its own advertised Capabilities, the calling Task
        # does (see benchy.core.Capabilities.accepts).
        default_text = "\n".join(
            part.text
            for message in request.messages
            for part in message.parts
            if isinstance(part, TextPart)
        )
        return finish(Response(text=default_text))

    async def aclose(self) -> None:
        self.closed = True


def load_echo(rest: str, **opts: Any) -> EchoSystem:
    """`echo:` loader — `rest` is an arbitrary free-form label, not parsed."""
    return EchoSystem(f"echo:{rest}", **opts)


def load_mock(rest: str, **opts: Any) -> EchoSystem:
    """`mock:` loader — identical to `echo:`, just a friendlier alias."""
    return EchoSystem(f"mock:{rest}", **opts)
