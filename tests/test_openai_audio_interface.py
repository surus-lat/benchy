"""Tests for OpenAIAudioInterface."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.interfaces.openai_audio_interface import OpenAIAudioInterface


def _make_interface() -> OpenAIAudioInterface:
    connection_info = {
        "base_url": "https://api.openai.com/v1",
        "provider_type": "openai_audio",
        "timeout": 30,
        "max_retries": 1,
        "max_concurrent": 2,
        "api_key": "sk-test",
    }
    return OpenAIAudioInterface(connection_info, "whisper-1")


def test_capabilities_advertise_audio_only():
    iface = _make_interface()
    caps = iface.capabilities
    assert caps.supports_audio is True
    assert caps.supports_multimodal is False
    assert caps.supports_schema is False
    assert caps.supports_files is False


def test_prepare_request_returns_expected_shape():
    iface = _make_interface()

    class Task:
        language = "es"

    req = iface.prepare_request(
        {"id": "x", "audio_path": "/tmp/audio.wav", "language": "pt"},
        Task(),
    )
    assert req == {
        "sample_id": "x",
        "audio_path": "/tmp/audio.wav",
        "language": "pt",
    }


def test_prepare_request_falls_back_to_task_language():
    iface = _make_interface()

    class Task:
        language = "es"

    req = iface.prepare_request(
        {"id": "x", "audio_path": "/tmp/audio.wav"},
        Task(),
    )
    assert req["language"] == "es"


def test_prepare_request_raises_when_audio_path_missing():
    iface = _make_interface()
    with pytest.raises(ValueError, match="audio_path"):
        iface.prepare_request({"id": "x"}, object())


@pytest.mark.asyncio
async def test_transcribe_single_returns_output_dict(tmp_path: Path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF....fakewav")

    iface = _make_interface()
    mock_create = AsyncMock(return_value="hola mundo")
    iface.client = MagicMock()
    iface.client.audio.transcriptions.create = mock_create

    result = await iface._transcribe_single(str(audio), "es", "sample-1")
    assert result["output"] == "hola mundo"
    assert result["raw"] == "hola mundo"
    assert result["error"] is None
    assert result["error_type"] is None
    mock_create.assert_awaited_once()
    kwargs = mock_create.await_args.kwargs
    assert kwargs["model"] == "whisper-1"
    assert kwargs["language"] == "es"
    assert kwargs["response_format"] == "text"
