"""Tests for benchy_cli_probe connection info construction."""

from argparse import Namespace

from src.benchy_cli_probe import _build_connection_info


def _probe_args(**overrides):
    defaults = {
        "model_name": "test-model",
        "provider": "openai",
        "base_url": None,
        "api_key": None,
        "api_key_env": None,
        "image_max_edge": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_probe_uses_provider_capability_defaults_for_anthropic() -> None:
    info = _build_connection_info(_probe_args(provider="anthropic", model_name="claude-3-5-sonnet"))

    assert info["capabilities"]["supports_multimodal"] is False
    assert info["capabilities"]["supports_schema"] is True
    assert info["capabilities"]["request_modes"] == ["chat"]


def test_probe_uses_provider_capability_defaults_for_alibaba() -> None:
    info = _build_connection_info(_probe_args(provider="alibaba", model_name="qwen-max"))

    assert info["capabilities"]["supports_multimodal"] is True
    assert info["capabilities"]["supports_schema"] is True
    assert info["capabilities"]["request_modes"] == ["chat"]
