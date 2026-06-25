"""Tests for the qwen3_asr_chat adapter — mocked, no real model load."""
import asyncio
import contextlib
import sys
import types
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch", reason="torch is required for qwen3_asr_chat adapter tests")


@contextlib.contextmanager
def _patched_qwen_asr(model):
    """Inject a fake `qwen_asr` module into sys.modules so the adapter's
    lazy import resolves to our MagicMock. Avoids importing the real
    package during tests."""
    qwen_asr_mod = types.ModuleType("qwen_asr")
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = model
    qwen_asr_mod.Qwen3ASRModel = fake_class
    original = sys.modules.get("qwen_asr")
    sys.modules["qwen_asr"] = qwen_asr_mod
    try:
        yield fake_class
    finally:
        if original is None:
            sys.modules.pop("qwen_asr", None)
        else:
            sys.modules["qwen_asr"] = original


def _fake_model(transcribe_return):
    m = MagicMock()
    m.transcribe.return_value = transcribe_return
    return m


def test_qwen3_asr_chat_returns_standard_result_shape():
    """One sample in → one {output, raw, error, error_type} out."""
    hyp = MagicMock()
    hyp.text = "olá mundo"
    fake = _fake_model([hyp])
    with _patched_qwen_asr(fake):
        from src.adapters.qwen3_asr_chat import Adapter

        adapter = Adapter(
            "Qwen/Qwen3-ASR-0.6B",
            {"dtype": "bfloat16", "device_map": "cpu"},
        )
        result = asyncio.run(
            adapter.generate_batch(
                [{"sample_id": "s1", "audio_path": "/tmp/f.wav", "language": "pt"}]
            )
        )

    assert result[0] == {
        "output": "olá mundo",
        "raw": "olá mundo",
        "error": None,
        "error_type": None,
    }


def test_qwen3_asr_chat_passes_iso_language_to_transcribe():
    """The adapter maps a sample's ISO code (e.g. 'es') to the human
    string Qwen3-ASR expects ('Spanish')."""
    hyp = MagicMock()
    hyp.text = "hola"
    fake = _fake_model([hyp])
    with _patched_qwen_asr(fake):
        from src.adapters.qwen3_asr_chat import Adapter

        adapter = Adapter("Qwen/Qwen3-ASR-0.6B", {})
        asyncio.run(adapter.generate_batch([{"audio_path": "/tmp/f.wav", "language": "es"}]))

    fake.transcribe.assert_called_once()
    kwargs = fake.transcribe.call_args.kwargs
    assert kwargs["language"] == "Spanish"


def test_qwen3_asr_chat_config_language_overrides_sample_language():
    """If the YAML pins `language: Portuguese`, the sample's ISO code is ignored."""
    hyp = MagicMock()
    hyp.text = "olá"
    fake = _fake_model([hyp])
    with _patched_qwen_asr(fake):
        from src.adapters.qwen3_asr_chat import Adapter

        adapter = Adapter("Qwen/Qwen3-ASR-0.6B", {"language": "Portuguese"})
        asyncio.run(adapter.generate_batch([{"audio_path": "/tmp/f.wav", "language": "es"}]))

    fake.transcribe.assert_called_once()
    assert fake.transcribe.call_args.kwargs["language"] == "Portuguese"


def test_qwen3_asr_chat_captures_per_request_errors():
    bad_model = MagicMock()
    bad_model.transcribe.side_effect = ValueError("bad audio")
    with _patched_qwen_asr(bad_model):
        from src.adapters.qwen3_asr_chat import Adapter

        adapter = Adapter("Qwen/Qwen3-ASR-0.6B", {})
        result = asyncio.run(
            adapter.generate_batch([{"audio_path": "/tmp/missing.wav"}])
        )

    assert result[0]["output"] == ""
    assert result[0]["error_type"] == "ValueError"


def test_qwen3_asr_chat_loads_model_lazily():
    hyp = MagicMock()
    hyp.text = "ok"
    fake = _fake_model([hyp])
    with _patched_qwen_asr(fake) as fake_class:
        from src.adapters.qwen3_asr_chat import Adapter

        adapter = Adapter("Qwen/Qwen3-ASR-0.6B", {})
        fake_class.from_pretrained.assert_not_called()

        asyncio.run(adapter.generate_batch([{"audio_path": "/tmp/f.wav"}]))

        fake_class.from_pretrained.assert_called_once()


def test_qwen3_asr_chat_capabilities_declares_audio():
    from src.adapters.qwen3_asr_chat import Adapter

    caps = Adapter.capabilities
    assert caps.supports_audio is True
