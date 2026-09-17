"""`hf:` — architecture routing. No network: family=... is passed explicitly
or transformers.AutoConfig.from_pretrained is monkeypatched.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

from benchy.core import LoadError
from benchy.system import load
from benchy.system.hf import _detect_family, known_families


class TestKnownFamilies:
    def test_lists_all_four_families(self):
        assert set(known_families()) == {
            "transformers_audio",
            "voxtral_chat",
            "qwen3_asr_chat",
            "canary_nemo",
        }


class TestLoadHfRequiresARepoId:
    def test_empty_rest_raises_load_error(self):
        with pytest.raises(LoadError):
            load("hf:")


class TestExplicitFamilySelection:
    def test_unknown_family_raises_load_error(self):
        with pytest.raises(LoadError, match="unknown family"):
            load("hf:some/repo", family="not-a-real-family")

    def test_explicit_family_bypasses_detection(self):
        def fake_pipeline(*args, **kwargs):
            raise AssertionError("should not build a real pipeline when one is injected")

        system = load(
            "hf:openai/whisper-large-v3-turbo",
            family="transformers_audio",
            pipeline=fake_pipeline,
        )
        assert system.url == "hf:openai/whisper-large-v3-turbo"
        assert system.capabilities.audio_in is True


class TestDetectFamily:
    def test_whisper_model_type_routes_to_transformers_audio(self, monkeypatch):
        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained",
            lambda repo_id, **kw: SimpleNamespace(model_type="whisper", architectures=[]),
        )
        assert _detect_family("openai/whisper-large-v3-turbo", trust_remote_code=False) == "transformers_audio"

    def test_voxtral_model_type_routes_to_voxtral_chat(self, monkeypatch):
        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained",
            lambda repo_id, **kw: SimpleNamespace(model_type="voxtral", architectures=[]),
        )
        assert _detect_family("mistralai/Voxtral-Mini-3B-2507", trust_remote_code=True) == "voxtral_chat"

    def test_voxtral_architecture_name_routes_to_voxtral_chat(self, monkeypatch):
        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained",
            lambda repo_id, **kw: SimpleNamespace(model_type="", architectures=["VoxtralForConditionalGeneration"]),
        )
        assert _detect_family("some/repo", trust_remote_code=True) == "voxtral_chat"

    def test_qwen3_asr_architecture_routes_to_qwen3_asr_chat(self, monkeypatch):
        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained",
            lambda repo_id, **kw: SimpleNamespace(model_type="", architectures=["Qwen3ASRForConditionalGeneration"]),
        )
        assert _detect_family("Qwen/Qwen3-ASR", trust_remote_code=False) == "qwen3_asr_chat"

    def test_config_lookup_failure_falls_back_to_canary_by_name(self, monkeypatch):
        def raiser(repo_id, **kw):
            raise ValueError("unrecognized model_type: fastconformer")

        monkeypatch.setattr("transformers.AutoConfig.from_pretrained", raiser)
        assert _detect_family("nvidia/canary-1b-flash", trust_remote_code=False) == "canary_nemo"

    def test_config_lookup_failure_for_unrelated_repo_raises_load_error(self, monkeypatch):
        def raiser(repo_id, **kw):
            raise ValueError("boom")

        monkeypatch.setattr("transformers.AutoConfig.from_pretrained", raiser)
        with pytest.raises(LoadError, match="family="):
            _detect_family("some/unknown-repo", trust_remote_code=False)

    def test_unrecognized_model_type_defaults_to_transformers_audio(self, monkeypatch):
        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained",
            lambda repo_id, **kw: SimpleNamespace(model_type="some-other-asr-arch", architectures=[]),
        )
        assert _detect_family("some/repo", trust_remote_code=False) == "transformers_audio"

    def test_load_hf_uses_detection_when_family_is_omitted(self, monkeypatch):
        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained",
            lambda repo_id, **kw: SimpleNamespace(model_type="whisper", architectures=[]),
        )
        system = load("hf:openai/whisper-large-v3-turbo", pipeline=lambda *a, **k: None)
        assert system.repo_id == "openai/whisper-large-v3-turbo"
