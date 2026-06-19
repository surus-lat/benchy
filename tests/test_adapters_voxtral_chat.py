"""Tests for the voxtral_chat adapter — mocked, no real model load."""
import asyncio
import contextlib
from unittest.mock import MagicMock, patch

import torch


def _fake_model():
    m = MagicMock()
    # Returns input_ids + 3 new tokens, total 6 tokens.
    m.generate.return_value = torch.tensor([[10, 20, 30, 40, 50, 60]])
    # Mock the .to() chain used to move model to device.
    m.to.return_value = m
    return m


def _fake_processor():
    p = MagicMock()
    p.apply_chat_template.return_value = {
        # input length 3 — generate() returned 6 — so 3 new tokens decoded.
        "input_ids": torch.tensor([[10, 20, 30]]),
    }
    p.batch_decode.return_value = ["hola mundo"]
    return p


@contextlib.contextmanager
def _patched_loaders(model, processor):
    """Patch all three transformers loaders the adapter calls during _load."""
    with patch(
        "transformers.AutoConfig.from_pretrained", return_value=MagicMock()
    ), patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
    ) as model_load, patch(
        "transformers.AutoProcessor.from_pretrained", return_value=processor
    ) as proc_load:
        yield model_load, proc_load


def test_voxtral_chat_returns_standard_result_shape():
    """One sample in → one {output, raw, error, error_type} out."""
    with _patched_loaders(_fake_model(), _fake_processor()):
        from src.adapters.voxtral_chat import Adapter

        adapter = Adapter(
            "mistralai/Voxtral-Mini-4B-Realtime-2602",
            {"torch_dtype": "float16", "trust_remote_code": True, "device": "cpu"},
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


def test_voxtral_chat_captures_per_request_errors():
    """A generate() that raises ends up in error/error_type, not propagated."""
    bad_model = MagicMock()
    bad_model.to.return_value = bad_model
    bad_model.generate.side_effect = RuntimeError("CUDA OOM")

    with _patched_loaders(bad_model, _fake_processor()):
        from src.adapters.voxtral_chat import Adapter

        adapter = Adapter(
            "mistralai/Voxtral-Mini-4B-Realtime-2602", {"device": "cpu"}
        )
        result = asyncio.run(
            adapter.generate_batch(
                [{"sample_id": "s1", "audio_path": "/tmp/f.wav"}]
            )
        )

    assert result[0]["output"] == ""
    assert result[0]["error_type"] == "RuntimeError"
    assert "CUDA OOM" in result[0]["error"]


def test_voxtral_chat_loads_model_lazily():
    """from_pretrained is not called until the first generate_batch."""
    with _patched_loaders(_fake_model(), _fake_processor()) as (model_load, proc_load):
        from src.adapters.voxtral_chat import Adapter

        adapter = Adapter(
            "mistralai/Voxtral-Mini-4B-Realtime-2602", {"device": "cpu"}
        )
        model_load.assert_not_called()
        proc_load.assert_not_called()

        asyncio.run(adapter.generate_batch([{"audio_path": "/tmp/f.wav"}]))

        model_load.assert_called_once()
        proc_load.assert_called_once()


def test_voxtral_chat_capabilities_declares_audio():
    """Adapter capabilities mark supports_audio True."""
    from src.adapters.voxtral_chat import Adapter

    caps = Adapter.capabilities
    assert caps.supports_audio is True
    assert caps.supports_multimodal is False
