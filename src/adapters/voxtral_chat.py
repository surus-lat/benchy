"""voxtral_chat adapter — Mistral Voxtral family ASR.

Voxtral models are speech-seq2seq audio→text transcribers. They live
in `transformers >= 5.13` and need `mistral-common` for the
processor. The processor takes a raw audio array (not a chat
template) and the model generates the transcription.

Adapter pipeline:
  1. librosa loads the audio at 16 kHz mono → numpy array
  2. processor(audio=array) → input_ids + audio features
  3. model.generate(**inputs)
  4. processor.batch_decode(outputs) → transcription string

Config knobs (read from `voxtral_chat:` block in the model YAML):
  trust_remote_code: bool (default True)
  torch_dtype: "float16" | "float32" | "bfloat16" (default "float16")
  device: "auto" | "cpu" | "cuda" | "mps" (default "auto")
  max_new_tokens: int (default 256)
  sampling_rate: int (default 16000)
"""

from __future__ import annotations

from typing import Any

from src.adapters.base import BaseAdapter
from src.engine.protocols import InterfaceCapabilities


_DTYPE_MAP = {
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "float32": "float32",
    "fp32": "float32",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Adapter(BaseAdapter):
    capabilities = InterfaceCapabilities(supports_audio=True)

    def __init__(self, model_name: str, config: dict[str, Any]):
        super().__init__(model_name, config)
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _load(self) -> None:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
        )

        dtype_name = _DTYPE_MAP[self.config.get("torch_dtype", "float16")]
        torch_dtype = getattr(torch, dtype_name)
        trust_remote_code = bool(self.config.get("trust_remote_code", True))
        device = _resolve_device(self.config.get("device", "auto"))

        # Pre-load the config explicitly with trust_remote_code so AutoModel
        # doesn't re-trigger the unknown-model_type rejection. Voxtral lives
        # in transformers >= 5.13 (was added after 4.57.6).
        hf_config = AutoConfig.from_pretrained(
            self.model_name, trust_remote_code=trust_remote_code
        )
        # Voxtral is a speech-seq2seq model (audio → text), not causal LM.
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            config=hf_config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        if device != "cpu":
            self._model = self._model.to(device)
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
        )
        self._device = device

    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self._model is None:
            self._load()

        import librosa

        max_new_tokens = int(self.config.get("max_new_tokens", 256))
        sampling_rate = int(self.config.get("sampling_rate", 16000))
        model_dtype = next(self._model.parameters()).dtype

        results = []
        for req in requests:
            try:
                audio_array, _ = librosa.load(
                    req["audio_path"], sr=sampling_rate, mono=True
                )
                inputs = self._processor(
                    audio=audio_array,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                )
                # Cast float feature tensors to the model's dtype (fp16/bf16)
                # so they match the model weights. Keep int tensors (input_ids,
                # attention_mask) untouched.
                inputs = {
                    k: (v.to(model_dtype) if hasattr(v, "dtype") and v.dtype.is_floating_point else v)
                    for k, v in inputs.items()
                }
                if self._device != "cpu":
                    inputs = {
                        k: v.to(self._device) if hasattr(v, "to") else v
                        for k, v in inputs.items()
                    }
                output_ids = self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
                input_len = inputs["input_ids"].shape[-1]
                generated = output_ids[:, input_len:]
                text = self._processor.batch_decode(
                    generated, skip_special_tokens=True
                )[0]
                results.append(
                    {"output": text, "raw": text, "error": None, "error_type": None}
                )
            except Exception as exc:
                results.append(
                    {
                        "output": "",
                        "raw": "",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                )
        return results
