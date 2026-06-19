"""Tests for BaseAdapter shape and defaults."""
import asyncio
import pytest

from src.adapters.base import BaseAdapter
from src.engine.protocols import InterfaceCapabilities


def test_base_adapter_generate_batch_is_abstract():
    """Calling generate_batch on BaseAdapter raises NotImplementedError."""
    adapter = BaseAdapter("fake-model", {})
    with pytest.raises(NotImplementedError):
        asyncio.run(adapter.generate_batch([{"id": "x"}]))


def test_base_adapter_prepare_request_is_identity():
    """Default prepare_request returns the sample unchanged."""
    adapter = BaseAdapter("fake-model", {})
    sample = {"id": "s1", "audio_path": "/tmp/f.wav"}
    assert adapter.prepare_request(sample, task=None) is sample


def test_base_adapter_stores_model_name_and_config():
    """Constructor stashes its arguments as instance attributes."""
    cfg = {"some_knob": True}
    adapter = BaseAdapter("hf/repo", cfg)
    assert adapter.model_name == "hf/repo"
    assert adapter.config is cfg


def test_subclass_can_declare_capabilities_class_attribute():
    """Subclasses set `capabilities` as a class attribute typed
    InterfaceCapabilities — same pattern existing Interfaces use."""

    class _SubAdapter(BaseAdapter):
        capabilities = InterfaceCapabilities(
            supports_audio=True,
        )

        async def generate_batch(self, requests):
            return [{"output": "ok", "raw": "ok", "error": None, "error_type": None}]

    sub = _SubAdapter("fake", {})
    assert sub.capabilities.supports_audio is True
    result = asyncio.run(sub.generate_batch([{"id": "x"}]))
    assert result[0]["output"] == "ok"
