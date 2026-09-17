"""`endpoint:` — a generic OpenAI-shaped HTTP endpoint.

This is how a user exposes *any* AI-system (including their own agent
behind a URL) per the vision. It reuses OpenAISystem's wire format wholesale
— only the base_url/model/default api_key differ.
"""

from __future__ import annotations

import json

import httpx
import pytest

from benchy.core import LoadError, Message, Request, TextPart
from benchy.system import load
from benchy.system.openai_system import OpenAISystem


class TestLoadEndpoint:
    def test_requires_a_base_url(self):
        with pytest.raises(LoadError):
            load("endpoint:")

    def test_base_url_is_the_full_rest_of_the_url(self):
        system = load("endpoint:https://host/v1")
        assert isinstance(system, OpenAISystem)
        assert system._base_url == "https://host/v1"

    def test_url_round_trips(self):
        system = load("endpoint:https://host/v1")
        assert system.url == "endpoint:https://host/v1"

    def test_model_defaults_when_not_given(self):
        system = load("endpoint:https://host/v1")
        assert system.model

    def test_model_kwarg_is_honoured(self):
        system = load("endpoint:https://host/v1", model="my-agent")
        assert system.model == "my-agent"

    def test_default_api_key_is_a_placeholder_not_required(self):
        # endpoint: targets are typically local/self-hosted; no key required.
        system = load("endpoint:https://host/v1")
        assert system._api_key


class TestEndpointInvokesLikeOpenAI:
    @pytest.mark.asyncio
    async def test_lowers_a_request_the_same_way_as_openai_scheme(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "my-agent",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi back"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )

        system = load(
            "endpoint:https://host/v1",
            model="my-agent",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        response = await system.invoke(Request(messages=(Message.text("user", "hi"),)))
        assert response.text == "hi back"
        body = json.loads(calls[-1].content)
        assert body["model"] == "my-agent"
