"""transformers_audio family — the default `hf:` route.

Any transformers `automatic-speech-recognition` pipeline: Whisper and every
other architecture `transformers.pipeline` already knows how to run. This
is the fallback family when `hf:`'s architecture detection doesn't match
one of the three specialized families.

`pipeline=` accepts a pre-built pipeline callable (dependency injection for
tests, and for callers who already built one); otherwise the pipeline is
built lazily, on first `invoke()`, from `transformers.pipeline`.
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import Any

from benchy.core import AudioPart, Capabilities, LoadError, Request, Response
from benchy.system.base import BaseSystem
from benchy.system.hf._device import resolve_device, resolve_dtype


class TransformersAudioSystem(BaseSystem):
    def __init__(
        self,
        repo_id: str,
        *,
        url: str | None = None,
        device: str = "auto",
        torch_dtype: str = "float32",
        chunk_length_s: float = 30,
        trust_remote_code: bool = False,
        language: str | None = None,
        timeout: float = 600.0,
        capabilities: Capabilities | None = None,
        pipeline: Any = None,
    ) -> None:
        super().__init__(
            url or f"hf:{repo_id}",
            capabilities or Capabilities(text_in=False, audio_in=True),
        )
        self.repo_id = repo_id
        self._device_opt = device
        self._dtype_opt = torch_dtype
        self._chunk_length_s = chunk_length_s
        self._trust_remote_code = trust_remote_code
        self.language = language
        self.timeout = timeout
        self._pipeline = pipeline

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise LoadError(f"hf: transformers is required for hf: systems: {exc}") from exc

        try:
            device = resolve_device(self._device_opt)
            torch_dtype = resolve_dtype(self._dtype_opt)
            self._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=self.repo_id,
                device=device,
                torch_dtype=torch_dtype,
                chunk_length_s=self._chunk_length_s,
                trust_remote_code=self._trust_remote_code,
            )
        except Exception as exc:
            raise LoadError(f"hf: failed to load {self.repo_id!r}: {exc}") from exc
        return self._pipeline

    def _audio_input(self, part: AudioPart) -> Any:
        if part.path:
            return part.path
        if part.data is not None:
            import soundfile as sf

            array, sr = sf.read(io.BytesIO(part.data), dtype="float32")
            return {"array": array, "sampling_rate": part.sample_rate or sr}
        raise LoadError("AudioPart must set data or path for hf: transformers_audio")

    async def invoke(self, request: Request) -> Response:
        start = time.perf_counter()
        pipe = self._ensure_pipeline()  # LoadError propagates: a setup/config failure

        audio_part = next(
            (p for m in request.messages for p in m.parts if isinstance(p, AudioPart)), None
        )
        if audio_part is None:
            return Response(
                error="transformers_audio: request has no AudioPart",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            audio_input = self._audio_input(audio_part)
            language = request.params.get("language") or self.language
            kwargs: dict[str, Any] = {}
            if language:
                kwargs["generate_kwargs"] = {"language": language, "task": "transcribe"}
            result = await asyncio.wait_for(
                asyncio.to_thread(pipe, audio_input, **kwargs), timeout=self.timeout
            )
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            return Response(
                text=text.strip(), latency_ms=(time.perf_counter() - start) * 1000
            )
        except Exception as exc:
            return Response(
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    async def aclose(self) -> None:
        self._pipeline = None


def load(repo_id: str, **opts: Any) -> TransformersAudioSystem:
    return TransformersAudioSystem(repo_id, **opts)
