"""Tests for the adapter routing branch in connection.py."""
from unittest.mock import patch

from src.adapters.base import BaseAdapter
from src.engine.connection import get_interface_for_provider


class _StubAdapter(BaseAdapter):
    async def generate_batch(self, requests):
        return [{"output": "stub", "raw": "stub", "error": None, "error_type": None}]


def test_adapter_field_routes_through_get_adapter():
    """When connection_info has an 'adapter' key, get_interface_for_provider
    returns get_adapter(...)'s result and skips the legacy provider path."""
    connection_info = {
        "adapter": "voxtral_chat",
        "voxtral_chat": {"trust_remote_code": True, "torch_dtype": "float16"},
        # Existing fields are present but should be ignored when adapter: is set.
        "provider_type": "transformers_audio",
    }
    expected_instance = _StubAdapter(
        "voxtral-x", {"trust_remote_code": True, "torch_dtype": "float16"}
    )
    with patch(
        "src.adapters.get_adapter", return_value=expected_instance
    ) as gated:
        # provider_type can be anything when adapter is set — it's not consulted.
        result = get_interface_for_provider(
            "transformers_audio", connection_info, "voxtral-x"
        )
    assert result is expected_instance
    gated.assert_called_once_with(
        "voxtral_chat",
        "voxtral-x",
        {"trust_remote_code": True, "torch_dtype": "float16"},
    )


def test_no_adapter_field_falls_through_to_legacy_routing():
    """When 'adapter' is absent, get_interface_for_provider does NOT touch
    get_adapter — the legacy provider routing handles it."""
    connection_info = {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    }
    with patch("src.adapters.get_adapter") as gated:
        try:
            get_interface_for_provider("openai", connection_info, "gpt-4o-mini")
        except Exception:
            # The legacy path may raise on missing fields in a unit test;
            # that's fine — we only care that get_adapter wasn't touched.
            pass
    gated.assert_not_called()


def test_voxtral_yaml_resolves_through_adapter_path():
    """Loading the voxtral YAML wires connection_info to the voxtral_chat adapter."""
    import yaml
    from pathlib import Path

    cfg_path = Path("configs/models/voxtral-mini-4b-transformers.yaml")
    model_config = yaml.safe_load(cfg_path.read_text())

    assert model_config.get("adapter") == "voxtral_chat", (
        "Task 6 should have set adapter: voxtral_chat in the YAML"
    )

    from src.adapters.voxtral_chat import Adapter as VoxtralAdapter

    connection_info = dict(model_config)
    result = get_interface_for_provider(
        "huggingface", connection_info, model_config["model"]["name"]
    )
    assert isinstance(result, VoxtralAdapter)


def test_qwen3_asr_yaml_resolves_through_adapter_path():
    """Same shape for Qwen3-ASR."""
    import yaml
    from pathlib import Path

    cfg_path = Path("configs/models/qwen3-asr-0.6b-transformers.yaml")
    model_config = yaml.safe_load(cfg_path.read_text())

    assert model_config.get("adapter") == "qwen3_asr_chat"

    from src.adapters.qwen3_asr_chat import Adapter as Qwen3Adapter

    connection_info = dict(model_config)
    result = get_interface_for_provider(
        "huggingface", connection_info, model_config["model"]["name"]
    )
    assert isinstance(result, Qwen3Adapter)
