"""The built-in provider adapter — outside the engine core, on the far side of A.10.

Nothing in `benchy`'s eight core modules imports this file. It exists because VISION
asks benchy to handle "all the different ai-system configurations under the hood", and
A.11 permits the runtime to select a reusable provider adapter for
`ai-system.type: model`. The engine still sees only `invoke(dict) -> dict`.

**Transport is `llm_client`**, SURUS's shared LLM client, so there is one HTTP layer
for every SURUS system rather than one per project. It is an optional dependency
(`pip install 'benchy[providers]'`), imported only when a model provider is actually
selected — an external adapter runs without it.

This file owns only what is specific to *benchmarking*: turning the program contract
into a request, and refusing anything that would make the measurement lie.

**One adapter, a table of endpoints.** A provider is an endpoint plus the name of its
credential, and any of them can be pointed elsewhere with `<PROVIDER>_BASE_URL`.

Anthropic models on Bedrock are the one exception to "everything speaks chat
completions": they do not serve that endpoint at all, and are reached through Converse,
where the output schema becomes a forced tool call. `llm_client` owns that translation;
this file only says *which* request shape applies, explicitly rather than letting a
hostname heuristic decide — a `BEDROCK_BASE_URL` pointing at a gateway would otherwise
silently get the wrong one. Claude on Bedrock must be named by its cross-region
inference profile (`us.anthropic.…`), the only form it is invocable under.

Five refusals, each tested:

- **No provider fallback.** A run evaluates one AI-system (`R = (B, AI)`); failing over
  to another model would score a mixture under one name.
- **No format fallback.** Retrying without the schema would score the system on an
  easier task than the benchmark declares.
- **No injected defaults.** `llm_client.call` defaults to `temperature=0.1` and
  `max_tokens=2000`; both are passed explicitly as whatever the benchmark said, or none.
- **No silently dropped parameters.** `llm_client` forwards anything beyond its named
  arguments as a nested `extra_body`, which chat-completions endpoints ignore without an
  error — verified live on Together. Such parameters are refused at setup.
- **No type coercion.** A model that returns `"121.00"` for a `float` produces an
  `invalid_output`, because the AI-system did not honour the program contract.

Credentials come from the environment, never from benchmark YAML (spec §11).
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
from collections.abc import Mapping
from pathlib import Path

from benchy import types
from benchy.data import resolve_within
from benchy.errors import BenchyError

__all__ = ["for_system", "OpenAIChat"]

#: provider -> (endpoint, credential variable). `{NAME}` in an endpoint is read from the
#: environment. Bedrock's OpenAI-compatible endpoint takes a Bedrock API key as a bearer
#: token under AWS's own variable name; it does not serve Claude, Nova or Llama.
_ENDPOINTS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "bedrock": ("https://bedrock-runtime.{AWS_REGION}.amazonaws.com/openai/v1", "AWS_BEARER_TOKEN_BEDROCK"),
}

#: The parameters `llm_client` delivers correctly on every endpoint profile.
_FORWARDED = ("temperature", "max_tokens")

#: Formats the model must produce, since JSON schema has no type for them (paper A.6).
_FORMATS = {
    "date": "a calendar date as YYYY-MM-DD",
    "time": "a time of day as HH:MM:SS",
    "datetime": "an RFC 3339 timestamp including a timezone offset",
}

_JSON_TYPES = {"int": "integer", "float": "number", "bool": "boolean"}

# A library does not print; the host decides. Keeps `benchy run`'s stderr for diagnostics.
logging.getLogger("llm_client").addHandler(logging.NullHandler())


def for_system(ir: Mapping, workspace: Path | str, env: Mapping[str, str] | None = None) -> object:
    """The provider adapter for this IR's AI-system, or a run setup error.

    `env` is injectable so tests need neither real credentials nor a real endpoint.
    """
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
    try:
        from llm_client import ProviderProfile, call
    except ImportError:
        raise BenchyError(
            "runtime", "adapter_not_bound",
            "model providers need the providers extra: pip install 'benchy[providers]'",
            ["ai-system", "provider"],
        ) from None
    return OpenAIChat(ir, Path(workspace), os.environ if env is None else env, call, ProviderProfile)


class OpenAIChat:
    """An OpenAI-compatible chat-completions AI-system, reached through `llm_client`.

    Everything that can be known to be unsupported is rejected in `__init__`, so a run
    fails at setup rather than producing a column of identical `execution_error`s.
    """

    def __init__(self, ir: Mapping, workspace: Path, env: Mapping[str, str], call, profiles=None) -> None:
        system = ir["ai-system"]
        provider = system["provider"]
        endpoint, credential = _ENDPOINTS[provider]
        try:
            base_url = env.get(f"{provider.upper()}_BASE_URL") or endpoint.format_map(env)
        except KeyError as missing:
            raise BenchyError(
                "runtime", "adapter_not_bound",
                f"{missing.args[0]} is not set; the {provider} endpoint needs it",
            ) from None
        if not env.get(credential):
            raise BenchyError(
                "runtime", "adapter_not_bound",
                f"{credential} is not set; credentials are runtime policy and never belong in "
                f"benchmark YAML",
            )

        parameters = dict(system.get("parameters") or {})
        dropped = sorted(set(parameters) - set(_FORWARDED))
        if dropped:
            raise BenchyError(
                "runtime", "adapter_not_bound",
                f"ai-system.parameters {', '.join(dropped)} would not reach the model: llm_client "
                f"sends only {' and '.join(_FORWARDED)} as request parameters, and anything else as "
                f"a nested extra_body that chat-completions endpoints silently ignore",
                ["ai-system", "parameters"],
            )

        self.call = call
        # Bedrock serves Anthropic models only through Converse. Stated, not inferred:
        # a BEDROCK_BASE_URL pointing at a gateway would defeat a hostname heuristic.
        self.profile = None
        if provider == "bedrock" and "anthropic" in system["model"].lower():
            self.profile = getattr(profiles, "BEDROCK_CONVERSE", None)
            if self.profile is None:
                raise BenchyError(
                    "runtime", "adapter_not_bound",
                    "Claude on Bedrock needs an llm_client with Bedrock Converse support "
                    "(surus-lat/llm-client#5); the installed one has none, and Anthropic models "
                    "do not serve Bedrock's chat-completions endpoint",
                    ["ai-system", "model"],
                )
        self.base_url = base_url.rstrip("/")
        self.api_key = env[credential]
        self.model = system["model"]
        self.temperature = parameters.get("temperature")
        self.max_tokens = parameters.get("max_tokens")
        self.timeout = float(env.get("BENCHY_TIMEOUT", "120"))
        self.input_schema = ir["program"]["input"]
        self.response_format = {
            "type": "json_schema",
            "json_schema": {"name": "program_output", "strict": True, "schema": _json_schema(ir["program"]["output"])},
        }
        _reject_unsupported(ir)
        self.prompt = _prompt(system, workspace)

    async def invoke(self, input_object: dict) -> object:
        # Only sent when set: an llm_client without the parameter still works for every
        # provider that needs no explicit profile.
        routing = {"profile": self.profile} if self.profile is not None else {}
        try:
            reply = await self.call(
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": self._content(input_object)},
                ],
                base_url=self.base_url,
                api_key=self.api_key,
                model_name=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=self.response_format,
                timeout=self.timeout,
                operation="benchy",
                **routing,
            )
        except Exception as exc:
            # llm_client raises its transport's HTTP error; the body is what says why.
            response = getattr(exc, "response", None)
            if response is None:
                raise
            raise BenchyError("runtime", "provider_error", f"{response.status_code}: {response.text[:500]}") from None

        # A budget that runs out mid-answer truncates the output. Checked live: the
        # provider reports finish_reason "length" and returns partial JSON, which would
        # otherwise be scored as the system's own malformed answer. llm_client only
        # surfaces finish_reason from the version that exposes it; before that, a
        # truncation is still null-scored and zero-contributing, just less well explained.
        if reply.get("finish_reason") == "length":
            raise BenchyError(
                "runtime", "provider_error",
                "the model hit its token limit before completing the output; raise max_tokens in "
                "ai-system.parameters (reasoning models spend the budget on reasoning first)",
            )
        text = reply["content"]
        if not text:
            raise BenchyError(
                "runtime", "provider_error",
                "the model returned no content; if it is a reasoning model, raise max_tokens in "
                "ai-system.parameters, since reasoning tokens are spent before the answer is written",
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Returning the raw text lets the engine record it as the prediction and
            # classify it as invalid_output, which says more than an exception would.
            return text

    def _content(self, input_object: dict, path: tuple[str, ...] = ()) -> list[dict]:
        """One content part per input leaf, named so the model knows what it is."""
        parts: list[dict] = []
        for name, child in types.at(self.input_schema, path)["fields"].items():
            value, here = input_object[name], path + (name,)
            if child["type"] == "object":
                parts.extend(self._content(value, here))
            elif child["type"] == "image":
                parts.append({"type": "text", "text": f"{'.'.join(here)}:"})
                parts.append({"type": "image_url", "image_url": {"url": _data_url(value)}})
            else:
                parts.append({"type": "text", "text": f"{'.'.join(here)}: {value}"})
        return parts


def _reject_unsupported(ir: Mapping) -> None:
    for side, refused, why in (
        ("input", ("audio", "document"), "which this adapter does not send; only text-valued fields and image are"),
        ("output", types.ARTIFACTS, "and a chat completion returns text, so it cannot produce an artifact;"),
    ):
        schema = ir["program"][side]
        for path in types.leaves(schema):
            kind = types.at(schema, path)["type"]
            if kind in refused:
                raise BenchyError(
                    "runtime", "adapter_not_bound",
                    f"{side} field {'.'.join(path)} is {kind}, {why} supported",
                    list(path),
                )


def _prompt(system: Mapping, workspace: Path) -> str:
    reference = system.get("prompt")
    if not reference:
        return "Answer with a JSON object matching the required schema."
    # Benchmark-owned paths resolve from the workspace root (amendment §2).
    root = workspace.resolve()
    path = resolve_within(reference, root, root, ["ai-system", "prompt"], phase="runtime")
    if not path.is_file():
        raise BenchyError("runtime", "adapter_not_bound", f"prompt file not found: {path}", ["ai-system", "prompt"])
    return path.read_text(encoding="utf-8")


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
