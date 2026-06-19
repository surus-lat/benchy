"""Tests for adapter discovery via importlib + pkgutil."""
import pytest

import src.adapters
from src.adapters import list_adapters, get_adapter
from src.adapters.base import BaseAdapter


def test_list_adapters_excludes_base_and_underscore(monkeypatch, tmp_path):
    """Files starting with `_` or named `base` are not adapters."""
    (tmp_path / "voxtral_chat.py").write_text("")
    (tmp_path / "_private.py").write_text("")
    (tmp_path / "base.py").write_text("")
    (tmp_path / "qwen3_asr_chat.py").write_text("")
    (tmp_path / "__init__.py").write_text("")
    monkeypatch.setattr(src.adapters, "__path__", [str(tmp_path)])
    assert list_adapters() == ["qwen3_asr_chat", "voxtral_chat"]


def test_get_adapter_unknown_name_raises_with_available_list():
    """A typo'd adapter name surfaces a helpful error message."""
    with pytest.raises(ValueError) as exc_info:
        get_adapter("not_a_real_adapter", "model-x", {})
    msg = str(exc_info.value)
    assert "not_a_real_adapter" in msg
    assert "Available:" in msg


def test_get_adapter_error_message_lists_existing_adapters():
    """Error message must include the actual available list (not a stub)."""
    with pytest.raises(ValueError) as exc_info:
        get_adapter("definitely_not_real", "model-x", {})
    assert str(list_adapters()) in str(exc_info.value)
