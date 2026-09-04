"""OpenAI-compatible API adapter — bare metal."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx

from benchy.core import Request, Response, TextPart


def openai_system(
    model: str,
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    system_prompt: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: float = 120.0,
):
    """Build an OpenAI-compatible System."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = base_url.rstrip("/")

    async def invoke(request: Request) -> Response:
        start = asyncio.get_event_loop().time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for msg in request.messages:
            content = _format_message(msg)
            if content:
                messages.append({"role": msg.role, "content": content})

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                )
                response.raise_for_status()
                data = response.json()

            content = data["choices"][0]["message"]["content"]
            latency = (asyncio.get_event_loop().time() - start) * 1000

            value = content
            if isinstance(content, str):
                stripped = content.strip()
                if stripped.startswith("{") or stripped.startswith("["):
                    try:
                        value = json.loads(stripped)
                    except json.JSONDecodeError:
                        pass

            return Response(text=content, data=value, raw=data, latency_ms=latency)

        except Exception as exc:
            return Response(error=str(exc))

    def _format_message(msg) -> str:
        parts = []
        for part in msg.parts:
            if isinstance(part, TextPart):
                parts.append(part.text)
        return "\n".join(parts)

    class _OpenAISystem:
        def __init__(self):
            self.url = f"openai://{model}"
            self.capabilities = None

        async def invoke(self, request: Request) -> Response:
            return await invoke(request)

        async def aclose(self) -> None:
            pass

    return _OpenAISystem()
