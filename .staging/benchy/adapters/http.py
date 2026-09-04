"""HTTP endpoint adapter — bare metal."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from benchy.core import Request, Response, TextPart


def http_system(
    url: str,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    body_template: str | None = None,
    response_path: str | None = None,
    timeout: float = 120.0,
):
    """Build an HTTP System from a URL."""

    async def invoke(request: Request) -> Response:
        start = asyncio.get_event_loop().time()

        try:
            body = _build_body(request, body_template)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(method, url, json=body, headers=headers or {})
                response.raise_for_status()
                data = response.json()

            value = _extract(data, response_path)
            latency = (asyncio.get_event_loop().time() - start) * 1000

            return Response(
                text=str(value) if not isinstance(value, str) else value,
                data=value,
                raw=data,
                latency_ms=latency,
            )

        except Exception as exc:
            return Response(error=str(exc))

    def _build_body(request: Request, template: str | None) -> dict[str, Any]:
        text = _extract_text(request)
        if template is None:
            return {"input": text, "id": "sample"}

        body_str = template.replace("{{input}}", text).replace("{{text}}", text)
        try:
            return json.loads(body_str)
        except json.JSONDecodeError:
            return {"input": text, "id": "sample"}

    def _extract_text(request: Request) -> str:
        for msg in request.messages:
            for part in msg.parts:
                if isinstance(part, TextPart):
                    return part.text
        return ""

    def _extract(data: Any, path: str | None) -> Any:
        if path is None:
            return data
        current = data
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                current = current[int(part)]
            else:
                return data
        return current

    class _HTTPSystem:
        def __init__(self):
            self.url = url
            self.capabilities = None

        async def invoke(self, request: Request) -> Response:
            return await invoke(request)

        async def aclose(self) -> None:
            pass

    return _HTTPSystem()
