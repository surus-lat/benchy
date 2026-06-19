"""Tests for the in-process transformers ASR interface.

These tests inject fake ``torch`` and ``transformers`` modules into
``sys.modules`` so they run without the heavy optional dependencies. The
focus is on contract: capability flags, request shape, language-kwarg
gating, timeout/error surfacing.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Fake module fixtures
# ---------------------------------------------------------------------------


class _FakeDtype:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover
        return f"FakeDtype({self.name})"


class _FakeBackendsMps:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeBackends:
    mps = _FakeBackendsMps()


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _PipelineRecorder:
    """Records calls so tests can assert against kwargs."""

    def __init__(self, return_value: Any = None):
        self.init_calls: List[Tuple[Tuple, Dict]] = []
        self.call_args: List[Tuple[Tuple, Dict]] = []
        self.raises: Optional[Exception] = None
        self._return_value = return_value if return_value is not None else {"text": "hola mundo"}

    def __call__(self, *args, **kwargs):
        # When invoked as transformers.pipeline(...), record then return self.
        if args and args[0] == "automatic-speech-recognition":
            self.init_calls.append((args, kwargs))
            return self
        # Otherwise it's the actual pipeline call site.
        self.call_args.append((args, kwargs))
        if self.raises is not None:
            raise self.raises
        return self._return_value


@pytest.fixture
def fake_modules(monkeypatch):
    """Install fake ``torch`` and ``transformers`` modules and yield helpers."""
    torch_module = types.ModuleType("torch")
    torch_module.float32 = _FakeDtype("float32")
    torch_module.float16 = _FakeDtype("float16")
    torch_module.bfloat16 = _FakeDtype("bfloat16")
    torch_module.cuda = _FakeCuda()
    torch_module.backends = _FakeBackends()

    transformers_module = types.ModuleType("transformers")
    pipeline_recorder = _PipelineRecorder()
    transformers_module.pipeline = pipeline_recorder

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    yield {
        "torch": torch_module,
        "transformers": transformers_module,
        "pipeline": pipeline_recorder,
    }


def _make_iface(**overrides):
    from src.interfaces.transformers_audio_interface import TransformersAudioInterface

    connection_info = {
        "provider_type": "transformers_audio",
        "timeout": 60,
        "max_retries": 1,
        "max_concurrent": 1,
        "device": "cpu",
        "torch_dtype": "float32",
        "chunk_length_s": 30,
    }
    connection_info.update(overrides)
    return TransformersAudioInterface(connection_info, overrides.pop("model_name", "openai/whisper-small"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_capabilities_advertise_audio_only(fake_modules):
    iface = _make_iface()
    caps = iface.capabilities
    assert caps.supports_audio is True
    assert caps.supports_multimodal is False
    assert caps.supports_schema is False
    assert caps.supports_files is False
    assert caps.supports_logprobs is False


def test_pipeline_initialized_with_provider_kwargs(fake_modules):
    _make_iface(device="cpu", torch_dtype="float16", chunk_length_s=15)
    init_calls = fake_modules["pipeline"].init_calls
    assert len(init_calls) == 1
    _args, kwargs = init_calls[0]
    assert kwargs["model"] == "openai/whisper-small"
    assert kwargs["device"] == "cpu"
    assert kwargs["chunk_length_s"] == 15
    assert kwargs["torch_dtype"].name == "float16"
    assert kwargs["trust_remote_code"] is False


def test_pipeline_init_passes_trust_remote_code_when_opted_in(fake_modules):
    _make_iface(trust_remote_code=True)
    _args, kwargs = fake_modules["pipeline"].init_calls[0]
    assert kwargs["trust_remote_code"] is True


def test_device_auto_falls_back_to_cpu_when_no_accelerator(fake_modules):
    _make_iface(device="auto")
    _args, kwargs = fake_modules["pipeline"].init_calls[0]
    assert kwargs["device"] == "cpu"


def test_prepare_request_returns_expected_shape(fake_modules):
    iface = _make_iface()

    class FakeTask:
        language = "es"
        answer_type = "freeform"

    req = iface.prepare_request(
        {"id": "s1", "audio_path": "/tmp/a.wav", "language": "es"}, FakeTask()
    )
    assert req == {"sample_id": "s1", "audio_path": "/tmp/a.wav", "language": "es"}


def test_prepare_request_falls_back_to_task_language(fake_modules):
    iface = _make_iface()

    class FakeTask:
        language = "pt"

    req = iface.prepare_request({"id": "s2", "audio_path": "/tmp/b.wav"}, FakeTask())
    assert req["language"] == "pt"


def test_prepare_request_raises_when_audio_path_missing(fake_modules):
    iface = _make_iface()
    with pytest.raises(ValueError, match="audio_path"):
        iface.prepare_request({"id": "s3", "expected": "hola"}, object())


def test_transcribe_single_passes_language_kwargs_when_supported(fake_modules):
    iface = _make_iface(supports_language_kwarg=True)
    asyncio.run(
        iface._transcribe_single(audio_path="/tmp/a.wav", language="es", sample_id="s1")
    )
    args, kwargs = fake_modules["pipeline"].call_args[0]
    assert args == ("/tmp/a.wav",)
    assert kwargs == {"generate_kwargs": {"language": "es", "task": "transcribe"}}


def test_transcribe_single_omits_language_when_unsupported(fake_modules):
    iface = _make_iface(supports_language_kwarg=False)
    asyncio.run(
        iface._transcribe_single(audio_path="/tmp/a.wav", language="es", sample_id="s1")
    )
    _args, kwargs = fake_modules["pipeline"].call_args[0]
    assert "generate_kwargs" not in kwargs


def test_transcribe_single_returns_transcript_text(fake_modules):
    iface = _make_iface()
    fake_modules["pipeline"]._return_value = {"text": "  hola mundo  "}
    result = asyncio.run(
        iface._transcribe_single(audio_path="/tmp/a.wav", language="es", sample_id="s1")
    )
    assert result == {"output": "hola mundo", "raw": "hola mundo", "error": None, "error_type": None}


def test_transcribe_single_surfaces_runtime_errors(fake_modules):
    iface = _make_iface()
    fake_modules["pipeline"].raises = RuntimeError("model crashed")
    result = asyncio.run(
        iface._transcribe_single(audio_path="/tmp/a.wav", language="es", sample_id="s1")
    )
    assert result["output"] is None
    assert result["raw"] is None
    assert result["error_type"] == "RuntimeError"
    assert "model crashed" in result["error"]


def test_generate_batch_returns_one_result_per_request(fake_modules):
    iface = _make_iface()
    fake_modules["pipeline"]._return_value = {"text": "hola"}
    requests = [
        {"sample_id": "s1", "audio_path": "/tmp/a.wav", "language": "es"},
        {"sample_id": "s2", "audio_path": "/tmp/b.wav", "language": "es"},
    ]
    results = asyncio.run(iface.generate_batch(requests))
    assert len(results) == 2
    assert all(r["output"] == "hola" for r in results)


def test_unsupported_dtype_raises(fake_modules):
    with pytest.raises(ValueError, match="Unsupported torch_dtype"):
        _make_iface(torch_dtype="int128")
