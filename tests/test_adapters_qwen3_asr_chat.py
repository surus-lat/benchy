"""Tests for the qwen3_asr_chat adapter — mocked, no real model load."""
import asyncio
import contextlib
from unittest.mock import MagicMock, patch

import torch


def _fake_model():
    m = MagicMock()
    m.generate.return_value = torch.tensor([[10, 20, 30, 40, 50, 60]])
    m.to.return_value = m
    return m


def _fake_processor():
    p = MagicMock()
    p.apply_chat_template.return_value = {
        "input_ids": torch.tensor([[10, 20, 30]]),
    }
    p.batch_decode.return_value = ["olá mundo"]
    return p


@contextlib.contextmanager
def _patched_loaders(model, processor):
    with patch(
        "transformers.AutoConfig.from_pretrained", return_value=MagicMock()
    ), patch(
        "transformers.AutoModelForSpeechSeq2Seq.from_pretrained", return_value=model
    ) as model_load, patch(
        "transformers.AutoProcessor.from_pretrained", return_value=processor
    ) as proc_load:
        yield model_load, proc_load


def test_qwen3_asr_chat_returns_standard_result_shape():
    with _patched_loaders(_fake_model(), _fake_processor()):
        from src.adapters.qwen3_asr_chat import Adapter

        adapter = Adapter(
            "Qwen/Qwen3-ASR-0.6B",
            {"torch_dtype": "float16", "trust_remote_code": True, "device": "cpu"},
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


def test_qwen3_asr_chat_captures_per_request_errors():
    bad_model = MagicMock()
    bad_model.to.return_value = bad_model
    bad_model.generate.side_effect = ValueError("audio file not found")

    with _patched_loaders(bad_model, _fake_processor()):
        from src.adapters.qwen3_asr_chat import Adapter

        adapter = Adapter("Qwen/Qwen3-ASR-0.6B", {"device": "cpu"})
        result = asyncio.run(
            adapter.generate_batch(
                [{"sample_id": "s1", "audio_path": "/tmp/missing.wav"}]
            )
        )

    assert result[0]["output"] == ""
    assert result[0]["error_type"] == "ValueError"


def test_qwen3_asr_chat_loads_model_lazily():
    with _patched_loaders(_fake_model(), _fake_processor()) as (model_load, proc_load):
        from src.adapters.qwen3_asr_chat import Adapter

        adapter = Adapter("Qwen/Qwen3-ASR-0.6B", {"device": "cpu"})
        model_load.assert_not_called()
        proc_load.assert_not_called()

        asyncio.run(adapter.generate_batch([{"audio_path": "/tmp/f.wav"}]))

        model_load.assert_called_once()
        proc_load.assert_called_once()


def test_qwen3_asr_chat_capabilities_declares_audio():
    from src.adapters.qwen3_asr_chat import Adapter

    caps = Adapter.capabilities
    assert caps.supports_audio is True
