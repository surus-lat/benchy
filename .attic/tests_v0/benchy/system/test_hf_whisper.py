"""transformers_audio family (the default `hf:` route) — any transformers
`automatic-speech-recognition` pipeline, Whisper included.

A fake pipeline callable is always injected via `pipeline=`, so these tests
never build a real transformers pipeline and never touch the network.
"""

from __future__ import annotations

import io

import pytest

sf = pytest.importorskip("soundfile")

from benchy.core import AudioPart, Message, Request
from benchy.system import load


def _sine_wav_bytes(seconds: float = 0.1, sr: int = 16000) -> bytes:
    import numpy as np

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wave = 0.1 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV")
    return buf.getvalue()


class _FakePipeline:
    def __init__(self, text: str = "hola"):
        self.text = text
        self.calls: list[tuple] = []

    def __call__(self, audio_input, **kwargs):
        self.calls.append((audio_input, kwargs))
        return {"text": self.text}


def _audio_req(part: AudioPart, **kw) -> Request:
    return Request(messages=(Message(role="user", parts=(part,)),), **kw)


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_path_based_audio_is_passed_straight_to_the_pipeline(self):
        fake = _FakePipeline("transcribed text")
        system = load("hf:openai/whisper-large-v3-turbo", family="transformers_audio", pipeline=fake)
        response = await system.invoke(_audio_req(AudioPart(path="/tmp/some.wav")))
        assert response.ok
        assert response.text == "transcribed text"
        assert fake.calls[0][0] == "/tmp/some.wav"

    @pytest.mark.asyncio
    async def test_bytes_based_audio_is_decoded_before_the_pipeline_call(self):
        fake = _FakePipeline("from bytes")
        system = load("hf:openai/whisper-large-v3-turbo", family="transformers_audio", pipeline=fake)
        wav_bytes = _sine_wav_bytes()
        response = await system.invoke(_audio_req(AudioPart(data=wav_bytes, mime="audio/wav")))
        assert response.text == "from bytes"
        audio_input = fake.calls[0][0]
        assert isinstance(audio_input, dict)
        assert "array" in audio_input and "sampling_rate" in audio_input

    @pytest.mark.asyncio
    async def test_language_param_becomes_generate_kwargs(self):
        fake = _FakePipeline()
        system = load("hf:openai/whisper-large-v3-turbo", family="transformers_audio", pipeline=fake)
        await system.invoke(_audio_req(AudioPart(path="/tmp/x.wav"), params={"language": "es"}))
        kwargs = fake.calls[0][1]
        assert kwargs["generate_kwargs"] == {"language": "es", "task": "transcribe"}

    @pytest.mark.asyncio
    async def test_latency_ms_is_populated(self):
        fake = _FakePipeline()
        system = load("hf:openai/whisper-large-v3-turbo", family="transformers_audio", pipeline=fake)
        response = await system.invoke(_audio_req(AudioPart(path="/tmp/x.wav")))
        assert response.latency_ms is not None


class TestCapabilities:
    def test_default_capabilities_are_audio_only(self):
        system = load("hf:openai/whisper-large-v3-turbo", family="transformers_audio", pipeline=_FakePipeline())
        assert system.capabilities.audio_in is True
        assert system.capabilities.text_in is False


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_missing_audio_part_is_a_response_error_not_a_raise(self):
        from benchy.core import TextPart

        system = load("hf:openai/whisper-large-v3-turbo", family="transformers_audio", pipeline=_FakePipeline())
        response = await system.invoke(Request(messages=(Message(role="user", parts=(TextPart("hi"),)),)))
        assert not response.ok

    @pytest.mark.asyncio
    async def test_pipeline_exception_becomes_a_response_error_not_a_raise(self):
        def boom(audio_input, **kwargs):
            raise RuntimeError("pipeline exploded")

        system = load("hf:openai/whisper-large-v3-turbo", family="transformers_audio", pipeline=boom)
        response = await system.invoke(_audio_req(AudioPart(path="/tmp/x.wav")))
        assert not response.ok
        assert "pipeline exploded" in response.error


class TestAclose:
    @pytest.mark.asyncio
    async def test_aclose_releases_the_pipeline(self):
        system = load("hf:openai/whisper-large-v3-turbo", family="transformers_audio", pipeline=_FakePipeline())
        await system.aclose()
        assert system._pipeline is None
