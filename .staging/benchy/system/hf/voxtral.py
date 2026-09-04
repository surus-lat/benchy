"""voxtral_chat family — Mistral Voxtral speech-seq2seq ASR.

Voxtral models are speech-seq2seq audio -> text transcribers, added to
`transformers` at 5.13 and dependent on `mistral-common` for the
processor. The processor takes a raw audio array (not a chat template);
`model.generate()` then produces the transcription.

`model=`/`processor=` accept pre-built objects (dependency injection for
tests and for callers who already loaded them); otherwise both are built
lazily, on first `invoke()`.
"""

from __future__ import annotations

import io
import time
from typing import Any

from benchy.core import AudioPart, Capabilities, LoadError, Request, Response
from benchy.system.base import BaseSystem
from benchy.system.hf._device import resolve_device, resolve_dtype


class VoxtralChatSystem(BaseSystem):
    def __init__(
        self,
        repo_id: str,
        *,
        url: str | None = None,
        device: str = "auto",
        torch_dtype: str = "float16",
        trust_remote_code: bool = True,
        max_new_tokens: int = 256,
        sampling_rate: int = 16000,
        capabilities: Capabilities | None = None,
        model: Any = None,
        processor: Any = None,
    ) -> None:
        super().__init__(
            url or f"hf:{repo_id}",
            capabilities or Capabilities(kind="model", text_in=False, audio_in=True),
        )
        self.repo_id = repo_id
        self._device_opt = device
        self._dtype_opt = torch_dtype
        self._trust_remote_code = trust_remote_code
        self.max_new_tokens = max_new_tokens
        self.sampling_rate = sampling_rate
        self._model = model
        self._processor = processor
        self._device = "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise LoadError(f"hf: voxtral_chat requires transformers>=5.13: {exc}") from exc

        device = resolve_device(self._device_opt)
        try:
            torch_dtype = resolve_dtype(self._dtype_opt)
            hf_config = AutoConfig.from_pretrained(self.repo_id, trust_remote_code=self._trust_remote_code)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.repo_id,
                config=hf_config,
                trust_remote_code=self._trust_remote_code,
                torch_dtype=torch_dtype,
            )
            if device != "cpu":
                model = model.to(device)
            processor = AutoProcessor.from_pretrained(self.repo_id, trust_remote_code=self._trust_remote_code)
        except ImportError as exc:
            raise LoadError(f"hf: voxtral_chat is missing a dependency (mistral-common?): {exc}") from exc
        except Exception as exc:
            raise LoadError(f"hf: failed to load voxtral model {self.repo_id!r}: {exc}") from exc
        self._model, self._processor, self._device = model, processor, device

    def _load_audio_array(self, part: AudioPart):
        import librosa

        if part.path:
            array, _ = librosa.load(part.path, sr=self.sampling_rate, mono=True)
            return array
        if part.data is not None:
            array, _ = librosa.load(io.BytesIO(part.data), sr=self.sampling_rate, mono=True)
            return array
        raise LoadError("AudioPart must set data or path for hf: voxtral_chat")

    async def invoke(self, request: Request) -> Response:
        start = time.perf_counter()
        self._ensure_loaded()  # LoadError propagates: a setup/config failure

        audio_part = next(
            (p for m in request.messages for p in m.parts if isinstance(p, AudioPart)), None
        )
        if audio_part is None:
            return Response(
                error="voxtral_chat: request has no AudioPart",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            array = self._load_audio_array(audio_part)
            inputs = self._processor(audio=array, sampling_rate=self.sampling_rate, return_tensors="pt")
            model_dtype = next(self._model.parameters()).dtype
            inputs = {
                k: (v.to(model_dtype) if hasattr(v, "dtype") and v.dtype.is_floating_point else v)
                for k, v in inputs.items()
            }
            if self._device != "cpu":
                inputs = {k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            input_len = inputs["input_ids"].shape[-1]
            generated = output_ids[:, input_len:]
            text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
            return Response(text=text, raw=text, latency_ms=(time.perf_counter() - start) * 1000)
        except Exception as exc:
            return Response(
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    async def aclose(self) -> None:
        self._model = None
        self._processor = None


def load(repo_id: str, **opts: Any) -> VoxtralChatSystem:
    return VoxtralChatSystem(repo_id, **opts)
