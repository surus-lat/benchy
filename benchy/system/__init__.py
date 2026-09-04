"""benchy.system — load any AI-system behind one contract: `System.invoke`.

    from benchy.system import load, register, schemes, BaseSystem, EchoSystem

    system = load("openai:gpt-5-mini", temperature=0)
    system = load("endpoint:https://host/v1/chat", model="my-model")
    system = load("hf:openai/whisper-large-v3-turbo")
    system = load("python:./my_agent.py:Agent")
    system = load("echo:", text="42")          # dependency-free test double

A raw model, a node (model + optimized prompt), a workflow (composed
models), and a tool-using agent are all the same kind of thing from here:
something that answers `await system.invoke(request) -> Response` and
advertises `system.capabilities`. Which framework loads which architecture,
which HTTP client is used, which subprocess is spawned — all of that is
plumbing hidden behind `load(url, **opts)`.

This module never imports `torch` or `transformers` at import time (nor
`openai`/`httpx`, though those are cheap) — every scheme loader imports its
heavy dependencies lazily, inside the function that actually needs them, so
`import benchy.system` stays instant regardless of what's installed.

Error-handling policy (binding on every scheme shipped here):

- `load(url, **opts)` raises `benchy.core.LoadError` for anything wrong
  *before* a System exists: an unknown/malformed URL, a missing required
  config value, a dependency that isn't installed for the requested `hf:`
  family, a `python:` target that can't be imported.
- Once a System exists, `invoke()` **never lets a provider/transport
  exception escape**. A failed HTTP call, a 5xx, a timeout, a malformed
  provider response, a missing optional dependency discovered only at
  call time — all of these come back as `Response(error=str(exc))`, never
  as a raised exception. `Response.ok` is how a caller checks for failure.
  This matches `benchy.core.Record.error` being a plain field rather than
  something a run loop has to catch exceptions to populate.
- The single deliberate exception: `echo:`'s `raises=` kwarg exists
  *specifically* to let a caller's own exception-handling path be tested;
  it does not weaken the policy above for the other four schemes.

Five schemes ship in this package: `echo:` (+ `mock:` alias), `openai:`,
`endpoint:`, `hf:`, `python:`. `register(scheme, loader)` adds — or
replaces — a scheme; `schemes()` lists what's currently registered.
"""

from __future__ import annotations

from benchy.system.base import BaseSystem
from benchy.system.echo import EchoSystem, load_echo, load_mock
from benchy.system.endpoint import load_endpoint
from benchy.system.hf import load_hf
from benchy.system.openai_system import OpenAISystem, load_openai
from benchy.system.python_loader import load_python
from benchy.system.registry import load, register, schemes

register("echo", load_echo)
register("mock", load_mock)
register("openai", load_openai)
register("endpoint", load_endpoint)
register("hf", load_hf)
register("python", load_python)

__all__ = [
    "load",
    "register",
    "schemes",
    "BaseSystem",
    "EchoSystem",
    "OpenAISystem",
]
