"""`endpoint:` — a generic OpenAI-shaped HTTP endpoint.

Per the vision, this is how a user exposes *any* AI-system as something
benchy can grade: put it behind an OpenAI-compatible `/v1/chat/completions`
route and point `endpoint:` at it — including a user's own agent, a vLLM
server, or a self-hosted model gateway. There is nothing endpoint-specific
here beyond the URL/default parsing: `OpenAISystem` (see `openai_system.py`)
does all the actual lowering and error handling.

URL grammar: `endpoint:<base_url>` — the scheme is split off on the first
`:` only, so `rest` is the full base URL including its own `https://`.
`model=` (default: `"default"`) and `api_key=` (default: `"EMPTY"`, since
self-hosted endpoints usually don't require one) are the two opinionated
defaults; every other `OpenAISystem` kwarg passes straight through.
"""

from __future__ import annotations

from typing import Any

from benchy.core import LoadError
from benchy.system.openai_system import OpenAISystem


def load_endpoint(rest: str, **opts: Any) -> OpenAISystem:
    if not rest:
        raise LoadError("endpoint: URL must include a base URL, e.g. 'endpoint:https://host/v1'")
    model = opts.pop("model", None) or "default"
    opts.setdefault("api_key", "EMPTY")
    return OpenAISystem(model, url=f"endpoint:{rest}", base_url=rest, **opts)
