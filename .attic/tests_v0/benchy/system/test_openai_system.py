"""`openai:` — lowering Request -> OpenAI-compatible wire payload.

No network: every test wires an `httpx.MockTransport` in as the SDK's
transport and asserts on the JSON body the SDK would have sent, or feeds
back a canned response to check parsing.
"""

from __future__ import annotations

import json

import httpx
import pytest

from benchy.core import (
    AudioPart,
    Capabilities,
    ImagePart,
    LoadError,
    Message,
    Request,
    Response,
    TextPart,
)
from benchy.system import load
from benchy.system.openai_system import OpenAISystem


def _chat_completion(text: str = "hi", *, prompt_tokens=10, completion_tokens=5) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5-mini",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _transcription(text: str = "hola mundo") -> dict:
    return {"text": text}


class _Capture:
    """A MockTransport handler that records every request and replays canned bodies."""

    def __init__(self, body: dict | None = None, *, status: int = 200):
        self.calls: list[httpx.Request] = []
        self.body = body if body is not None else _chat_completion()
        self.status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return httpx.Response(self.status, json=self.body)

    @property
    def last_json(self) -> dict:
        return json.loads(self.calls[-1].content)


def _system(capture: _Capture, **kwargs) -> OpenAISystem:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(capture))
    return OpenAISystem(
        "gpt-5-mini",
        base_url="http://test/v1",
        api_key="sk-test",
        http_client=http_client,
        **kwargs,
    )


def _req(*parts, role="user", output_schema=None, meta=None, params=None) -> Request:
    return Request(
        messages=(Message(role=role, parts=tuple(parts)),),
        output_schema=output_schema,
        meta=meta or {},
        params=params or {},
    )


class TestLoadOpenAI:
    def test_requires_a_model(self):
        with pytest.raises(LoadError):
            load("openai:")

    def test_default_capabilities(self):
        system = load("openai:gpt-5-mini")
        assert system.capabilities.image_in
        assert system.capabilities.audio_in
        assert system.capabilities.structured_output

    def test_url_carries_the_model(self):
        system = load("openai:gpt-5-mini")
        assert system.url == "openai:gpt-5-mini"


class TestLoweringText:
    @pytest.mark.asyncio
    async def test_single_text_part_lowers_to_a_plain_string(self):
        capture = _Capture()
        system = _system(capture)
        await system.invoke(_req(TextPart("hello")))
        body = capture.last_json
        assert body["model"] == "gpt-5-mini"
        assert body["messages"] == [{"role": "user", "content": "hello"}]

    @pytest.mark.asyncio
    async def test_multiple_messages_preserve_role_order(self):
        capture = _Capture()
        system = _system(capture)
        request = Request(
            messages=(
                Message.text("system", "be terse"),
                Message.text("user", "hi"),
            )
        )
        await system.invoke(request)
        roles = [m["role"] for m in capture.last_json["messages"]]
        assert roles == ["system", "user"]


class TestLoweringMultimodal:
    @pytest.mark.asyncio
    async def test_multiple_text_parts_become_content_blocks(self):
        capture = _Capture()
        system = _system(capture)
        await system.invoke(_req(TextPart("a"), TextPart("b")))
        content = capture.last_json["messages"][0]["content"]
        assert content == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]

    @pytest.mark.asyncio
    async def test_image_part_with_bytes_becomes_a_data_uri(self):
        capture = _Capture()
        system = _system(capture)
        await system.invoke(_req(TextPart("what is this"), ImagePart(data=b"\x89PNG", mime="image/png")))
        content = capture.last_json["messages"][0]["content"]
        image_block = next(b for b in content if b["type"] == "image_url")
        assert image_block["image_url"]["url"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_image_part_with_url_passes_through_unchanged(self):
        capture = _Capture()
        system = _system(capture)
        await system.invoke(_req(ImagePart(url="https://example.com/x.png")))
        content = capture.last_json["messages"][0]["content"]
        image_block = next(b for b in content if b["type"] == "image_url")
        assert image_block["image_url"]["url"] == "https://example.com/x.png"

    @pytest.mark.asyncio
    async def test_audio_part_embedded_as_input_audio_block_when_chat_shaped(self):
        capture = _Capture()
        system = _system(capture)
        request = _req(
            TextPart("transcribe and summarize"),
            AudioPart(data=b"RIFF....", mime="audio/wav"),
            output_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
        )
        await system.invoke(request)
        content = capture.last_json["messages"][0]["content"]
        audio_block = next(b for b in content if b["type"] == "input_audio")
        assert audio_block["input_audio"]["format"] == "wav"
        assert audio_block["input_audio"]["data"]  # base64 present


class TestTranscriptionRouting:
    @pytest.mark.asyncio
    async def test_audio_only_request_with_no_schema_routes_to_transcriptions_endpoint(self):
        capture = _Capture(body=_transcription("hola mundo"))
        system = _system(capture)
        response = await system.invoke(_req(AudioPart(data=b"RIFF....", mime="audio/wav")))
        assert capture.calls[-1].url.path.endswith("/audio/transcriptions")
        assert response.text == "hola mundo"

    @pytest.mark.asyncio
    async def test_meta_mode_chat_forces_chat_even_with_audio(self):
        capture = _Capture()
        system = _system(capture)
        await system.invoke(
            _req(AudioPart(data=b"RIFF....", mime="audio/wav"), meta={"mode": "chat"})
        )
        assert capture.calls[-1].url.path.endswith("/chat/completions")

    @pytest.mark.asyncio
    async def test_meta_mode_transcription_forces_transcription_even_with_schema(self):
        capture = _Capture(body=_transcription("x"))
        system = _system(capture)
        await system.invoke(
            _req(
                AudioPart(data=b"RIFF....", mime="audio/wav"),
                output_schema={"type": "object"},
                meta={"mode": "transcription"},
            )
        )
        assert capture.calls[-1].url.path.endswith("/audio/transcriptions")


class TestStructuredOutput:
    @pytest.mark.asyncio
    async def test_output_schema_sets_response_format_json_schema_strict(self):
        capture = _Capture(body=_chat_completion('{"total": 1}'))
        system = _system(capture)
        schema = {"type": "object", "properties": {"total": {"type": "integer"}}}
        response = await system.invoke(_req(TextPart("extract"), output_schema=schema))
        body = capture.last_json
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
        assert response.data == {"total": 1}

    @pytest.mark.asyncio
    async def test_structured_output_false_never_sends_response_format(self):
        capture = _Capture()
        system = _system(capture, capabilities=Capabilities(structured_output=False))
        schema = {"type": "object", "properties": {"total": {"type": "integer"}}}
        await system.invoke(_req(TextPart("extract"), output_schema=schema))
        assert "response_format" not in capture.last_json


class TestUsageAndLatency:
    @pytest.mark.asyncio
    async def test_usage_is_filled_from_the_response(self):
        capture = _Capture(body=_chat_completion(prompt_tokens=7, completion_tokens=3))
        system = _system(capture)
        response = await system.invoke(_req(TextPart("hi")))
        assert response.usage.input_tokens == 7
        assert response.usage.output_tokens == 3

    @pytest.mark.asyncio
    async def test_latency_ms_is_populated_and_positive(self):
        capture = _Capture()
        system = _system(capture)
        response = await system.invoke(_req(TextPart("hi")))
        assert response.latency_ms is not None
        assert response.latency_ms >= 0


class TestParams:
    @pytest.mark.asyncio
    async def test_temperature_and_max_tokens_are_sent(self):
        # A non-gpt-5-style model: temperature and plain max_tokens both apply.
        capture = _Capture(body=_chat_completion())
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(capture))
        system = OpenAISystem(
            "gpt-4o-mini",
            base_url="http://test/v1",
            api_key="sk-test",
            http_client=http_client,
            temperature=0.2,
            max_tokens=64,
        )
        await system.invoke(_req(TextPart("hi")))
        body = capture.last_json
        assert body["temperature"] == 0.2
        assert body["max_tokens"] == 64

    @pytest.mark.asyncio
    async def test_gpt5_family_uses_max_completion_tokens_and_skips_temperature(self):
        capture = _Capture()
        system = _system(capture, max_tokens=64)
        await system.invoke(_req(TextPart("hi")))
        body = capture.last_json
        assert body.get("max_completion_tokens") == 64
        assert "temperature" not in body
        assert "max_tokens" not in body


class TestErrorHandlingPolicy:
    @pytest.mark.asyncio
    async def test_a_5xx_becomes_a_response_with_error_not_a_raise(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": {"message": "boom"}})

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        system = OpenAISystem(
            "gpt-5-mini", base_url="http://test/v1", api_key="sk-test", http_client=http_client, max_retries=0
        )
        response = await system.invoke(_req(TextPart("hi")))
        assert not response.ok
        assert response.error

    @pytest.mark.asyncio
    async def test_a_connection_error_becomes_a_response_with_error_not_a_raise(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        system = OpenAISystem(
            "gpt-5-mini", base_url="http://test/v1", api_key="sk-test", http_client=http_client, max_retries=0
        )
        response = await system.invoke(_req(TextPart("hi")))
        assert not response.ok
        assert response.error  # the SDK wraps it (e.g. APIConnectionError); just must not raise


class TestAclose:
    @pytest.mark.asyncio
    async def test_aclose_closes_the_underlying_client(self):
        capture = _Capture()
        system = _system(capture)
        await system.invoke(_req(TextPart("hi")))  # forces client creation
        await system.aclose()
        # openai's AsyncOpenAI exposes is_closed after .close()
        assert system._client.is_closed
