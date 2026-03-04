"""Tests for probe runner inline eval path."""

import pytest

from src.probe import runner


@pytest.mark.asyncio
async def test_run_probe_for_eval_wraps_mode_and_schema_checks_with_timeout(monkeypatch):
    seen_checks = []

    async def fake_run_check_with_timeout(check_name, check_func, timeout, fail_mode):
        seen_checks.append(check_name)
        if check_name == "access_readiness":
            return {"status": "passed"}
        if check_name == "request_modes":
            return {"chat": {"tested": True, "ok": True}}
        if check_name == "schema_transports":
            return {"response_format": {"tested": True, "ok": True}}
        return {"status": fail_mode}

    monkeypatch.setattr(runner, "_run_check_with_timeout", fake_run_check_with_timeout)

    report = await runner.run_probe_for_eval(
        connection_info={
            "provider_type": "openai",
            "base_url": "https://api.openai.com/v1",
            "capabilities": {
                "request_modes": ["chat"],
                "supports_schema": True,
            },
        },
        model_name="gpt-4o-mini",
    )

    assert "access_readiness" in seen_checks
    assert "request_modes" in seen_checks
    assert "schema_transports" in seen_checks
    assert report["modes"]["chat"]["ok"] is True
    assert report["schema_transports"]["response_format"]["ok"] is True


@pytest.mark.asyncio
async def test_run_probe_for_eval_records_request_mode_timeout_result(monkeypatch):
    async def fake_run_check_with_timeout(check_name, check_func, timeout, fail_mode):
        if check_name == "access_readiness":
            return {"status": "passed"}
        if check_name == "request_modes":
            return {"status": "failed", "error": "Timeout after 30s", "evidence": {}}
        return {"status": fail_mode}

    monkeypatch.setattr(runner, "_run_check_with_timeout", fake_run_check_with_timeout)

    report = await runner.run_probe_for_eval(
        connection_info={
            "provider_type": "openai",
            "base_url": "https://api.openai.com/v1",
            "capabilities": {
                "request_modes": ["chat"],
                "supports_schema": False,
            },
        },
        model_name="gpt-4o-mini",
    )

    assert report["checks"]["request_modes"]["status"] == "failed"
    assert report["modes"]["chat"]["tested"] is False
