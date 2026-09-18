"""voxtral_chat family — Mistral Voxtral speech-seq2seq models.

Real `torch` and `librosa` are used (both are in the dev venv) to build
believable fake `model`/`processor` objects and synthetic audio, so these
tests exercise the real tensor-shuffling logic in `invoke()` without ever
downloading a real Voxtral checkpoint or needing `mistral-common`.
"""

from __future__ import annotations

import io

import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")

from benchy.core import AudioPart, Message, Request  # noqa: E402
from benchy.system import load  # noqa: E402


def _sine_wav_bytes(seconds: float = 0.1, sr: int = 16000) -> bytes:
    import numpy as np

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wave = 0.05 * np.sin(2 * np.pi * 220 * t)
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV")
    return buf.getvalue()


class _FakeProcessor:
    def __call__(self, audio, sampling_rate, return_tensors="pt"):
        input_ids = torch.ones((1, 4), dtype=torch.long)
        input_features = torch.zeros((1, 4, 8), dtype=torch.float32)
        return {"input_ids": input_ids, "input_features": input_features}

    def batch_decode(self, generated, skip_special_tokens=True):
        return ["hola desde voxtral"]


class _FakeModel:
    def parameters(self):
        yield torch.zeros(1, dtype=torch.float32)

    def generate(self, **inputs):
        input_len = inputs["input_ids"].shape[-1]
        # pretend it generated 2 extra tokens beyond the prompt
        return torch.ones((1, input_len + 2), dtype=torch.long)


def _audio_req(part: AudioPart) -> Request:
    return Request(messages=(Message(role="user", parts=(part,)),))


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_injected_model_and_processor_transcribe(self, tmp_path):
        wav_path = tmp_path / "a.wav"
        wav_path.write_bytes(_sine_wav_bytes())
        system = load(
            "hf:mistralai/Voxtral-Mini-3B-2507",
            family="voxtral_chat",
            model=_FakeModel(),
            processor=_FakeProcessor(),
        )
        response = await system.invoke(_audio_req(AudioPart(path=str(wav_path))))
        assert response.ok
        assert response.text == "hola desde voxtral"

    @pytest.mark.asyncio
    async def test_bytes_based_audio_also_works(self):
        system = load(
            "hf:mistralai/Voxtral-Mini-3B-2507",
            family="voxtral_chat",
            model=_FakeModel(),
            processor=_FakeProcessor(),
        )
        response = await system.invoke(
            _audio_req(AudioPart(data=_sine_wav_bytes(), mime="audio/wav"))
        )
        assert response.text == "hola desde voxtral"


class TestCapabilities:
    def test_default_capabilities_are_audio_only(self):
        system = load(
            "hf:mistralai/Voxtral-Mini-3B-2507",
            family="voxtral_chat",
            model=_FakeModel(),
            processor=_FakeProcessor(),
        )
        assert system.capabilities.audio_in is True


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_missing_audio_part_is_a_response_error(self):
        from benchy.core import TextPart

        system = load(
            "hf:mistralai/Voxtral-Mini-3B-2507",
            family="voxtral_chat",
            model=_FakeModel(),
            processor=_FakeProcessor(),
        )
        response = await system.invoke(Request(messages=(Message(role="user", parts=(TextPart("hi"),)),)))
        assert not response.ok

    @pytest.mark.asyncio
    async def test_inference_exception_becomes_a_response_error(self, tmp_path):
        class BoomModel(_FakeModel):
            def generate(self, **inputs):
                raise RuntimeError("cuda oom")

        wav_path = tmp_path / "a.wav"
        wav_path.write_bytes(_sine_wav_bytes())
        system = load(
            "hf:mistralai/Voxtral-Mini-3B-2507",
            family="voxtral_chat",
            model=BoomModel(),
            processor=_FakeProcessor(),
        )
        response = await system.invoke(_audio_req(AudioPart(path=str(wav_path))))
        assert not response.ok
        assert "cuda oom" in response.error


class TestMissingDependencyDegradesToLoadError:
    @pytest.mark.asyncio
    async def test_a_failure_while_resolving_the_hf_config_becomes_a_load_error(self, monkeypatch, tmp_path):
        def raiser(repo_id, **kw):
            raise ImportError("mistral_common is not installed")

        monkeypatch.setattr("transformers.AutoConfig.from_pretrained", raiser)
        wav_path = tmp_path / "a.wav"
        wav_path.write_bytes(_sine_wav_bytes())
        system = load("hf:mistralai/Voxtral-Mini-3B-2507", family="voxtral_chat")
        with pytest.raises(Exception) as excinfo:
            await system.invoke(_audio_req(AudioPart(path=str(wav_path))))
        from benchy.core import LoadError

        assert isinstance(excinfo.value, LoadError)
        assert "mistral_common" in str(excinfo.value)
