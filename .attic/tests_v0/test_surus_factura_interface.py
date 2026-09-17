"""Tests for SURUS Factura interface request fallback and response parsing."""

import httpx
import pytest

from src.interfaces.surus.surus_factura_interface import SurusFacturaInterface


def _build_interface(monkeypatch) -> SurusFacturaInterface:
    monkeypatch.setenv("SURUS_API_KEY", "test-key")
    config = {
        "surus_factura": {
            "endpoint": "https://example.test/factura",
            "api_key_env": "SURUS_API_KEY",
        }
    }
    return SurusFacturaInterface(config, model_name="surus-factura-test", provider_type="surus_factura")


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        if not self._responses:
            raise AssertionError("No fake responses left")
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_make_request_retries_with_image_url_only_for_field_errors(monkeypatch):
    interface = _build_interface(monkeypatch)

    first = httpx.Response(
        422,
        json={"detail": "Unknown field: image"},
        request=httpx.Request("POST", interface.endpoint),
    )
    second = httpx.Response(
        200,
        json={"data": {"ok": True}},
        request=httpx.Request("POST", interface.endpoint),
    )
    client = _FakeClient([first, second])

    response = await interface._make_request_with_client(
        client,
        {"image_path": "https://example.test/invoice.jpg", "sample_id": "s1"},
    )

    assert response == {"data": {"ok": True}}
    assert len(client.calls) == 2
    assert "image" in client.calls[0]["json"]
    assert "image_url" in client.calls[1]["json"]


@pytest.mark.asyncio
async def test_make_request_skips_image_url_retry_for_non_field_errors(monkeypatch):
    interface = _build_interface(monkeypatch)

    response = httpx.Response(
        422,
        json={"detail": "Invalid base64 payload"},
        request=httpx.Request("POST", interface.endpoint),
    )
    client = _FakeClient([response])

    with pytest.raises(httpx.HTTPStatusError):
        await interface._make_request_with_client(
            client,
            {"image_path": "https://example.test/invoice.jpg", "sample_id": "s2"},
        )

    assert len(client.calls) == 1
    assert "image" in client.calls[0]["json"]


@pytest.mark.asyncio
async def test_make_request_skips_retry_for_payload_error_even_with_field_mention(monkeypatch):
    interface = _build_interface(monkeypatch)

    response = httpx.Response(
        422,
        json={"detail": "Invalid base64 payload for field image"},
        request=httpx.Request("POST", interface.endpoint),
    )
    client = _FakeClient([response])

    with pytest.raises(httpx.HTTPStatusError):
        await interface._make_request_with_client(
            client,
            {"image_path": "https://example.test/invoice.jpg", "sample_id": "s3"},
        )

    assert len(client.calls) == 1
    assert "image" in client.calls[0]["json"]


def test_parse_response_rejects_non_dict(monkeypatch):
    interface = _build_interface(monkeypatch)

    result = interface._parse_response(["not", "a", "dict"])

    assert result["output"] is None
    assert result["error_type"] == "invalid_response"
    assert "Unexpected response format" in (result["error"] or "")


def test_parse_response_supports_dict_choices_content(monkeypatch):
    interface = _build_interface(monkeypatch)

    result = interface._parse_response(
        {"choices": [{"message": {"content": {"field": "value"}}}]}
    )

    assert result["output"] == {"field": "value"}
    assert result["error"] is None


def test_parse_response_handles_invalid_json_choices_content(monkeypatch):
    interface = _build_interface(monkeypatch)

    result = interface._parse_response(
        {"choices": [{"message": {"content": "not-json"}}]}
    )

    assert result["output"] is None
    assert result["error_type"] == "invalid_response"
