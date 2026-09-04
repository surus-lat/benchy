"""canary_nemo family — NVIDIA Canary via the NeMo toolkit.

The "missing dependency degrades to a clear LoadError" test below forces
`import nemo...` to fail via `sys.modules` rather than relying on the
package's actual absence — `nemo-toolkit` happens to be installed in this
project's venv (transitively, for other reasons), and letting a "package
missing" test fall through to the real loader would silently start
downloading a real checkpoint over the network. The happy-path tests
inject a fake model to exercise the request/response mapping without the
real dependency either way.
"""

from __future__ import annotations

import sys

import pytest

from benchy.core import AudioPart, LoadError, Message, Request, TextPart
from benchy.system import load


class _FakeHypothesis:
    def __init__(self, text: str):
        self.text = text


class _FakeCanaryModel:
    def __init__(self, text: str = "hola mundo"):
        self.text = text
        self.calls: list[dict] = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return [_FakeHypothesis(self.text)]


def _audio_req(path: str) -> Request:
    return Request(messages=(Message(role="user", parts=(AudioPart(path=path),)),))


class TestMissingDependencyDegradesCleanly:
    @pytest.mark.asyncio
    async def test_nemo_not_installed_raises_a_clear_load_error(self, monkeypatch):
        # None in sys.modules is Python's documented way to force an
        # ImportError for a specific module, regardless of whether it's
        # actually installed — see importlib docs.
        monkeypatch.setitem(sys.modules, "nemo", None)
        monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", None)
        system = load("hf:nvidia/canary-1b-flash", family="canary_nemo")
        with pytest.raises(LoadError, match="nemo-toolkit"):
            await system.invoke(_audio_req("/tmp/a.wav"))


class TestHappyPathWithInjectedModel:
    @pytest.mark.asyncio
    async def test_transcribe_maps_hypothesis_text_into_response(self):
        fake = _FakeCanaryModel("hola desde canary")
        system = load(
            "hf:nvidia/canary-1b-flash", family="canary_nemo", model=fake, source_lang="es"
        )
        response = await system.invoke(_audio_req("/tmp/a.wav"))
        assert response.ok
        assert response.text == "hola desde canary"
        assert fake.calls[0]["source_lang"] == "es"

    @pytest.mark.asyncio
    async def test_target_lang_defaults_to_source_lang(self):
        fake = _FakeCanaryModel()
        system = load(
            "hf:nvidia/canary-1b-flash", family="canary_nemo", model=fake, source_lang="fr"
        )
        await system.invoke(_audio_req("/tmp/a.wav"))
        assert fake.calls[0]["target_lang"] == "fr"


class TestCapabilities:
    def test_default_capabilities_are_audio_only(self):
        system = load("hf:nvidia/canary-1b-flash", family="canary_nemo", model=_FakeCanaryModel())
        assert system.capabilities.audio_in is True


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_missing_audio_part_is_a_response_error(self):
        system = load("hf:nvidia/canary-1b-flash", family="canary_nemo", model=_FakeCanaryModel())
        response = await system.invoke(Request(messages=(Message(role="user", parts=(TextPart("hi"),)),)))
        assert not response.ok

    @pytest.mark.asyncio
    async def test_missing_audio_path_is_a_response_error(self):
        system = load("hf:nvidia/canary-1b-flash", family="canary_nemo", model=_FakeCanaryModel())
        response = await system.invoke(
            Request(messages=(Message(role="user", parts=(AudioPart(data=b"x"),)),))
        )
        assert not response.ok

    @pytest.mark.asyncio
    async def test_transcribe_exception_becomes_a_response_error(self):
        class BoomModel(_FakeCanaryModel):
            def transcribe(self, **kwargs):
                raise RuntimeError("nemo blew up")

        system = load("hf:nvidia/canary-1b-flash", family="canary_nemo", model=BoomModel())
        response = await system.invoke(_audio_req("/tmp/a.wav"))
        assert not response.ok
        assert "nemo blew up" in response.error
