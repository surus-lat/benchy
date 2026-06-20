"""Tests for the canary_nemo adapter — mocked, no real model load."""
import asyncio
import contextlib
import sys
import types
from unittest.mock import MagicMock, patch


@contextlib.contextmanager
def _patched_nemo(model):
    """Patch nemo.collections.asr.models.EncDecMultiTaskModel.from_pretrained
    by injecting a fake `nemo` module tree into sys.modules. Avoids actually
    importing NeMo (heavy) during tests."""
    # Build the module chain nemo.collections.asr.models.
    nemo_mod = types.ModuleType("nemo")
    collections_mod = types.ModuleType("nemo.collections")
    asr_mod = types.ModuleType("nemo.collections.asr")
    models_mod = types.ModuleType("nemo.collections.asr.models")
    fake_class = MagicMock()
    fake_class.from_pretrained.return_value = model
    models_mod.EncDecMultiTaskModel = fake_class
    nemo_mod.collections = collections_mod
    collections_mod.asr = asr_mod
    asr_mod.models = models_mod

    originals = {}
    for name, mod in [
        ("nemo", nemo_mod),
        ("nemo.collections", collections_mod),
        ("nemo.collections.asr", asr_mod),
        ("nemo.collections.asr.models", models_mod),
    ]:
        originals[name] = sys.modules.get(name)
        sys.modules[name] = mod
    try:
        yield fake_class
    finally:
        for name, orig in originals.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


def _fake_model(transcribe_return):
    m = MagicMock()
    m.transcribe.return_value = transcribe_return
    # cfg.decoding.beam.beam_size mutation path
    m.cfg = MagicMock()
    m.cfg.decoding = MagicMock()
    return m


def test_canary_nemo_returns_standard_result_shape():
    """One sample in → one {output, raw, error, error_type} out."""
    hyp = MagicMock()
    hyp.text = "hola mundo"
    fake = _fake_model([hyp])
    with _patched_nemo(fake):
        from src.adapters.canary_nemo import Adapter

        adapter = Adapter(
            "nvidia/canary-1b-flash",
            {"device": "cpu", "source_lang": "es"},
        )
        result = asyncio.run(
            adapter.generate_batch(
                [{"sample_id": "s1", "audio_path": "/tmp/f.wav", "language": "es"}]
            )
        )

    assert len(result) == 1
    assert result[0] == {
        "output": "hola mundo",
        "raw": "hola mundo",
        "error": None,
        "error_type": None,
    }


def test_canary_nemo_captures_per_request_errors():
    """A transcribe() that raises ends up in error/error_type, not propagated."""
    fake = MagicMock()
    fake.transcribe.side_effect = RuntimeError("audio not found")
    fake.cfg = MagicMock()
    fake.cfg.decoding = MagicMock()
    with _patched_nemo(fake):
        from src.adapters.canary_nemo import Adapter

        adapter = Adapter("nvidia/canary-1b-flash", {"device": "cpu"})
        result = asyncio.run(
            adapter.generate_batch([{"sample_id": "s1", "audio_path": "/tmp/missing.wav"}])
        )

    assert result[0]["output"] == ""
    assert result[0]["error_type"] == "RuntimeError"
    assert "audio not found" in result[0]["error"]


def test_canary_nemo_loads_model_lazily():
    """from_pretrained is not called until the first generate_batch."""
    hyp = MagicMock()
    hyp.text = "ok"
    fake = _fake_model([hyp])
    with _patched_nemo(fake) as fake_class:
        from src.adapters.canary_nemo import Adapter

        adapter = Adapter("nvidia/canary-1b-flash", {"device": "cpu"})
        fake_class.from_pretrained.assert_not_called()

        asyncio.run(adapter.generate_batch([{"audio_path": "/tmp/f.wav"}]))

        fake_class.from_pretrained.assert_called_once()


def test_canary_nemo_capabilities_declares_audio():
    from src.adapters.canary_nemo import Adapter

    caps = Adapter.capabilities
    assert caps.supports_audio is True


def test_canary_nemo_falls_back_to_str_when_no_text_attr():
    """Older NeMo Hypothesis objects may not expose .text — fall back to str()."""
    fake = _fake_model(["plain string transcription"])
    with _patched_nemo(fake):
        from src.adapters.canary_nemo import Adapter

        adapter = Adapter("nvidia/canary-1b-flash", {"device": "cpu"})
        result = asyncio.run(adapter.generate_batch([{"audio_path": "/tmp/f.wav"}]))

    assert result[0]["output"] == "plain string transcription"
    assert result[0]["error"] is None
