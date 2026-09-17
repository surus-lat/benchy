"""Provider adapters — outside the engine core, on the far side of A.10's boundary.

Nothing in `benchy`'s eight core modules imports this file. It exists because VISION
asks benchy to handle "all the different ai-system configurations under the hood", and
A.11 permits the runtime to select a reusable provider adapter for
`ai-system.type: model`. The engine still sees only `invoke(dict) -> dict`.

**One adapter, not many.** An OpenAI-compatible `/v1/chat/completions` endpoint
reaches OpenAI, Together, vLLM, LM Studio, Ollama's compatibility mode, the hosted
aggregators, any self-hosted gateway, and a user's own agent behind a route. A
per-vendor adapter reaches one vendor. So there is one adapter here and a table of
endpoints: a provider is a default base URL plus the name of its credential, both
overridable, and nothing else.

Two deliberate refusals:

- **No type coercion.** A model that returns `"121.00"` for a `float` produces an
  `invalid_output`, and that is the correct measurement — the AI-system did not honour
  the program contract. Quietly repairing it would make the benchmark lie.
- **No SDK.** Transport is `urllib`, so installing benchy does not install a vendor
  package. The engine's dependency is PyYAML; this file adds nothing. One consequence
  is worth knowing: `urllib` announces itself as `Python-urllib/x.y`, which sits on
  Cloudflare's default block list — Together returns 403 (code 1010) for it. Hence the
  explicit `User-Agent` below. Found by running against the real endpoint; no local
  stand-in would have shown it.

Credentials come from the environment, never from benchmark YAML (spec §11).
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from benchy import types
from benchy.errors import BenchyError

__all__ = ["for_system", "OpenAIChat"]

#: Formats the model must produce, since JSON schema has no type for them (paper A.6).
_FORMATS = {
    "date": "a calendar date as YYYY-MM-DD",
    "time": "a time of day as HH:MM:SS",
    "datetime": "an RFC 3339 timestamp including a timezone offset",
}

_JSON_TYPES = {"int": "integer", "float": "number", "bool": "boolean"}

#: Anything but urllib's default, which providers' WAFs block. See the module docstring.
_USER_AGENT = "benchy/1.0"

#: A provider is its default endpoint. Credentials and overrides derive from the name
#: (`together` -> `TOGETHER_API_KEY`, `TOGETHER_BASE_URL`), so adding one is one line.
_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "together": "https://api.together.xyz/v1",
}


def for_system(ir: Mapping, workspace: Path | str, env: Mapping[str, str] | None = None) -> object:
    """The provider adapter for this IR's AI-system, or a run setup error.

    `env` is injectable so tests need neither real credentials nor a real endpoint.
    """
    env = os.environ if env is None else env
    system = ir["ai-system"]
    if system.get("type") != "model":
        raise BenchyError(
            "runtime", "adapter_not_bound",
            f"ai-system.type is {system.get('type')!r}; only 'model' resolves to a built-in "
            f"provider adapter. Bind an adapter explicitly for an external AI-system.",
            ["ai-system"],
        )
    if system["provider"] not in _ENDPOINTS:
        raise BenchyError(
            "runtime", "adapter_not_bound",
            f"no built-in adapter for provider {system['provider']!r}; benchy ships "
            f"{', '.join(sorted(_ENDPOINTS))}, and each reaches any OpenAI-compatible "
            f"endpoint via <PROVIDER>_BASE_URL",
            ["ai-system", "provider"],
        )
    return OpenAIChat(ir, Path(workspace), env)


class OpenAIChat:
    """An OpenAI-compatible chat-completions AI-system.

    Everything that can be known to be unsupported is rejected in `__init__`, so a run
    fails at setup rather than producing a column of identical `execution_error`s.
    """

    def __init__(self, ir: Mapping, workspace: Path, env: Mapping[str, str]) -> None:
        self.model = ir["ai-system"]["model"]
        self.parameters = dict(ir["ai-system"].get("parameters") or {})
        self.input_schema = ir["program"]["input"]
        self.schema = _json_schema(ir["program"]["output"])
        provider = ir["ai-system"]["provider"]
        prefix = provider.upper()
        self.base_url = env.get(f"{prefix}_BASE_URL", _ENDPOINTS[provider]).rstrip("/")
        self.timeout = float(env.get("BENCHY_TIMEOUT", "120"))

        key = env.get(f"{prefix}_API_KEY")
        if not key:
            raise BenchyError(
                "runtime", "adapter_not_bound",
                f"{prefix}_API_KEY is not set; credentials are runtime policy and never "
                f"belong in benchmark YAML",
            )
        self.api_key = key

        self._reject_unsupported(ir)
        self.prompt = self._load_prompt(ir, workspace)

    def _reject_unsupported(self, ir: Mapping) -> None:
        for path in types.leaves(ir["program"]["input"]):
            kind = types.at(ir["program"]["input"], path)["type"]
            if kind in ("audio", "document"):
                raise BenchyError(
                    "runtime", "adapter_not_bound",
                    f"input field {'.'.join(path)} is {kind}, which this adapter does not "
                    f"send; only text-valued fields and image are supported",
                    list(path),
                )
        for path in types.leaves(ir["program"]["output"]):
            kind = types.at(ir["program"]["output"], path)["type"]
            if kind in types.ARTIFACTS:
                raise BenchyError(
                    "runtime", "adapter_not_bound",
                    f"output field {'.'.join(path)} is {kind}; a chat completion returns "
                    f"text, so it cannot produce an artifact",
                    list(path),
                )

    def _load_prompt(self, ir: Mapping, workspace: Path) -> str:
        reference = ir["ai-system"].get("prompt")
        if not reference:
            return "Answer with a JSON object matching the required schema."
        # Benchmark-owned paths resolve from the workspace root (amendment §2).
        from benchy.data import resolve_within

        path = resolve_within(reference, workspace.resolve(), workspace.resolve(),
                              ["ai-system", "prompt"], phase="runtime")
        if not path.is_file():
            raise BenchyError("runtime", "adapter_not_bound", f"prompt file not found: {path}",
                              ["ai-system", "prompt"])
        return path.read_text(encoding="utf-8")

    async def invoke(self, input_object: dict) -> object:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": self._content(input_object)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "program_output", "strict": True, "schema": self.schema},
            },
            **self.parameters,
        }
        body = await asyncio.to_thread(self._post, payload)
        choice = body["choices"][0]
        text = choice["message"].get("content")

        # A reasoning model spends completion tokens on thinking before it writes
        # anything, so too small a budget returns reasoning and no answer. Saying so
        # beats letting it surface as an inscrutable invalid_output.
        if choice.get("finish_reason") == "length":
            raise BenchyError(
                "runtime", "provider_error",
                "the model hit its token limit before completing the output; raise "
                "max_tokens in ai-system.parameters (reasoning models spend the budget "
                "on reasoning_content first)",
            )
        if not text:
            raise BenchyError(
                "runtime", "provider_error",
                "the model returned no content"
                + (" (only reasoning_content)" if choice["message"].get("reasoning_content") else ""),
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Returning the raw text lets the engine record it as the prediction and
            # classify it as invalid_output, which says more than an exception would.
            return text

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                # urllib's default UA is blocked by some providers' WAF; see above.
                "User-Agent": _USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise BenchyError("runtime", "provider_error",
                              f"{exc.code} {exc.reason}: {exc.read()[:500].decode(errors='replace')}") from None

    def _content(self, input_object: dict, path: tuple[str, ...] = ()) -> list[dict]:
        """One content part per input leaf, named so the model knows what it is."""
        parts: list[dict] = []
        node = types.at(self.input_schema, path)
        for name, child in node["fields"].items():
            value, here = input_object[name], path + (name,)
            if child["type"] == "object":
                parts.extend(self._content(value, here))
            elif child["type"] == "image":
                parts.append({"type": "text", "text": f"{'.'.join(here)}:"})
                parts.append({"type": "image_url", "image_url": {"url": _data_url(value)}})
            else:
                parts.append({"type": "text", "text": f"{'.'.join(here)}: {value}"})
        return parts


def _data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}"


def _json_schema(node: Mapping) -> dict:
    """The output IR as a strict JSON schema, for structured outputs.

    Types JSON schema cannot express — dates, times, timestamps — become strings
    carrying their canonical form in `description`, which is the only channel the
    model has for it.
    """
    kind = node["type"]
    if kind == "object":
        return {
            "type": "object",
            "properties": {name: _json_schema(child) for name, child in node["fields"].items()},
            "required": list(node["fields"]),
            "additionalProperties": False,
        }
    if kind == "enum":
        return {"type": "string", "enum": list(node["values"])}
    if kind in _JSON_TYPES:
        return {"type": _JSON_TYPES[kind]}
    if kind in _FORMATS:
        return {"type": "string", "description": _FORMATS[kind]}
    return {"type": "string"}
