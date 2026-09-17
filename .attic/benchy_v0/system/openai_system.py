"""`openai:` — OpenAI-compatible chat completions (+ transcriptions).

Lowers a transport-free `benchy.core.Request` into the OpenAI chat wire
format: `TextPart` -> a text content block (or a bare string when it's the
message's only part), `ImagePart` -> an `image_url` block (a data URI when
`data`/`path` bytes are given, the URL as-is when only `url` is set),
`AudioPart` -> an `input_audio` block. `Request.output_schema` becomes
native structured output (`response_format: json_schema`, strict mode) when
`capabilities.structured_output` is True; when it's False the schema is
silently ignored here — encoding it into the prompt instead is the Task's
job (see `benchy.core.Capabilities` docstring), not this System's.

A request is **transcription-shaped** — routed to
`/audio/transcriptions` instead of `/chat/completions` — when it carries an
`AudioPart` and no `output_schema`. `request.meta["mode"]` overrides the
heuristic explicitly: `"chat"` forces the chat path even with audio,
`"transcription"` forces the transcription path even with a schema.

Backs both `openai:` and `endpoint:` (`endpoint.py` is a two-line wrapper
that points this same class at a caller-chosen `base_url`) — a generic
OpenAI-shaped HTTP endpoint is exactly an OpenAI system with a different
host, per the vision's "exposing any ai-system as an endpoint" clause.

Error-handling policy (see `benchy/system/__init__.py`): every exception
raised by the OpenAI SDK or the transport underneath it — HTTP errors,
timeouts, connection failures, malformed JSON — is caught in `invoke()` and
turned into `Response(error=...)`. Nothing escapes `invoke()` as a raised
exception. `load_openai()` raises `LoadError` for a missing model id.

`openai` and `httpx` are imported lazily (inside `_get_client`), never at
module import time, so `import benchy.system` stays cheap.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from benchy.core import (
    AudioPart,
    Capabilities,
    ImagePart,
    LoadError,
    Message,
    Request,
    Response,
    SystemFailure,
    TextPart,
    Usage,
)
from benchy.system.base import BaseSystem

# Models whose chat/completions endpoint wants `max_completion_tokens`
# instead of `max_tokens`, and rejects a non-default `temperature`.
_GPT5_STYLE_MARKERS = ("gpt-5", "o1", "o3", "o4")


def _image_data_uri(part: ImagePart) -> str:
    if part.url:
        return part.url
    data = part.data
    if data is None and part.path:
        data = Path(part.path).read_bytes()
    if data is None:
        raise LoadError("ImagePart must set one of data, path, or url")
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{part.mime};base64,{b64}"


def _audio_format(mime: str) -> str:
    return mime.split("/", 1)[-1] if "/" in mime else mime


def _audio_bytes(part: AudioPart) -> bytes:
    if part.data is not None:
        return part.data
    if part.path:
        return Path(part.path).read_bytes()
    raise LoadError(
        "AudioPart must set data or path for the openai:/endpoint: schemes "
        "(fetching a bare url is not supported here)"
    )


def _lower_parts(parts: tuple[Any, ...]) -> str | list[dict[str, Any]]:
    if len(parts) == 1 and isinstance(parts[0], TextPart):
        return parts[0].text
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            blocks.append({"type": "image_url", "image_url": {"url": _image_data_uri(part)}})
        elif isinstance(part, AudioPart):
            blocks.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(_audio_bytes(part)).decode("ascii"),
                        "format": _audio_format(part.mime),
                    },
                }
            )
        else:  # pragma: no cover - core.Part is a closed union
            raise LoadError(f"openai: unsupported content part {part!r}")
    return blocks


def _lower_messages(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": _lower_parts(m.parts)} for m in messages]


def _is_transcription_shaped(request: Request) -> bool:
    mode = request.meta.get("mode")
    if mode == "chat":
        return False
    if mode == "transcription":
        return True
    has_audio = any(isinstance(p, AudioPart) for m in request.messages for p in m.parts)
    return has_audio and request.output_schema is None


def _sanitize_schema_strict(schema: Any) -> Any:
    """Coerce a JSON Schema fragment towards OpenAI strict-mode's rules:
    every object needs `additionalProperties: false` and every property
    listed in `required`. Applied recursively through `properties`/`items`.
    """
    if not isinstance(schema, dict):
        return schema
    schema = dict(schema)
    props = schema.get("properties")
    if isinstance(props, dict):
        schema["properties"] = {k: _sanitize_schema_strict(v) for k, v in props.items()}
        schema.setdefault("required", list(props.keys()))
        schema.setdefault("additionalProperties", False)
    items = schema.get("items")
    if isinstance(items, dict):
        schema["items"] = _sanitize_schema_strict(items)
    return schema


class OpenAISystem(BaseSystem):
    """OpenAI-compatible chat/completions + transcriptions System."""

    def __init__(
        self,
        model: str,
        *,
        url: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_tokens_param: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        structured_output: bool = True,
        capabilities: Capabilities | None = None,
        client: Any = None,
        http_client: Any = None,
    ) -> None:
        super().__init__(
            url or f"openai:{model}",
            capabilities
            if capabilities is not None
            else Capabilities(image_in=True, audio_in=True, structured_output=structured_output),
        )
        self.model = model
        self._base_url = base_url
        self._api_key = api_key
        self._api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._max_tokens_param = max_tokens_param
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = client
        self._http_client = http_client

    def _is_gpt5_style(self) -> bool:
        lowered = self.model.lower()
        return any(marker in lowered for marker in _GPT5_STYLE_MARKERS)

    def _max_tokens_key(self) -> str:
        if self._max_tokens_param:
            return self._max_tokens_param
        return "max_completion_tokens" if self._is_gpt5_style() else "max_tokens"

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import os

        from openai import AsyncOpenAI

        api_key = self._api_key or os.getenv(self._api_key_env) or "EMPTY"
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=api_key,
            max_retries=0,  # benchy's engine layer owns retries, not the SDK
            http_client=self._http_client,
            timeout=self.timeout,
        )
        return self._client

    async def invoke(self, request: Request) -> Response:
        start = time.perf_counter()
        try:
            client = self._get_client()
            if _is_transcription_shaped(request):
                response = await self._invoke_transcription(client, request)
            else:
                response = await self._invoke_chat(client, request)
        except Exception as exc:  # never let a provider/transport error escape invoke()
            return Response(
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        if response.latency_ms is None:
            response = dataclasses.replace(response, latency_ms=(time.perf_counter() - start) * 1000)
        return response

    async def _invoke_chat(self, client: Any, request: Request) -> Response:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": _lower_messages(request.messages),
            "timeout": self.timeout,
        }
        if not self._is_gpt5_style():
            params["temperature"] = self.temperature
        params[self._max_tokens_key()] = self.max_tokens
        params.update(request.params)

        use_structured = request.output_schema is not None and self.capabilities.structured_output
        if use_structured:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "benchy_output",
                    "strict": True,
                    "schema": _sanitize_schema_strict(dict(request.output_schema)),
                },
            }

        completion = await client.chat.completions.create(**params)
        choice = completion.choices[0]
        content = choice.message.content

        usage_obj = getattr(completion, "usage", None)
        usage = None
        if usage_obj is not None:
            usage = Usage(
                input_tokens=getattr(usage_obj, "prompt_tokens", None),
                output_tokens=getattr(usage_obj, "completion_tokens", None),
            )

        data = None
        if use_structured and content:
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                data = None

        return Response(text=content, data=data, usage=usage)

    async def _invoke_transcription(self, client: Any, request: Request) -> Response:
        audio_part = next(
            (p for m in request.messages for p in m.parts if isinstance(p, AudioPart)),
            None,
        )
        if audio_part is None:
            raise SystemFailure("transcription-shaped request has no AudioPart")
        data = _audio_bytes(audio_part)
        filename = f"audio.{_audio_format(audio_part.mime)}"
        kwargs: dict[str, Any] = {
            "model": self.model,
            "file": (filename, data, audio_part.mime),
            "timeout": self.timeout,
        }
        language = request.params.get("language")
        if language:
            kwargs["language"] = language
        transcription = await client.audio.transcriptions.create(**kwargs)
        text = getattr(transcription, "text", None)
        if text is None:
            text = str(transcription)
        return Response(text=text.strip())

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()


def load_openai(rest: str, **opts: Any) -> OpenAISystem:
    if not rest:
        raise LoadError("openai: URL must include a model id, e.g. 'openai:gpt-5-mini'")
    return OpenAISystem(rest, url=f"openai:{rest}", **opts)
