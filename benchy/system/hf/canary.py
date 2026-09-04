"""canary_nemo family — NVIDIA Canary ASR via the NeMo toolkit.

Canary models (`nvidia/canary-1b-flash` and friends) are FastConformer
encoder-decoder models that only load through NVIDIA's NeMo toolkit, not
`transformers` — there's no standard HF `model_type` for them, which is
why `hf:`'s architecture detection can't find them by config (see
`hf/__init__.py`'s `_detect_family` fallback: a `"canary"` substring match
on the repo id, but pass `family="canary_nemo"` explicitly for clarity).

`nemo-toolkit[asr]` is an optional extra (`benchy[transcription]`), not a
core dependency. Constructing a `canary_nemo` system always succeeds
(loading is lazy); when the package isn't installed, the first `invoke()`
raises `LoadError` naming it — a documented degradation rather than a bare
`ImportError` leaking out. Pass `model=` to inject an already-loaded NeMo
model (or a fake, for tests).
"""

from __future__ import annotations

import time
from typing import Any

from benchy.core import AudioPart, Capabilities, LoadError, Request, Response
from benchy.system.base import BaseSystem
from benchy.system.hf._device import resolve_device


class CanaryNemoSystem(BaseSystem):
    def __init__(
        self,
        repo_id: str,
        *,
        url: str | None = None,
        device: str = "auto",
        source_lang: str = "es",
        target_lang: str | None = None,
        pnc: str = "yes",
        beam_size: int = 1,
        timestamps: bool = False,
        batch_size: int = 1,
        capabilities: Capabilities | None = None,
        model: Any = None,
    ) -> None:
        super().__init__(
            url or f"hf:{repo_id}",
            capabilities or Capabilities(text_in=False, audio_in=True),
        )
        self.repo_id = repo_id
        self._device_opt = device
        self.source_lang = source_lang
        self.target_lang = target_lang or source_lang
        self.pnc = pnc
        self.beam_size = beam_size
        self.timestamps = timestamps
        self.batch_size = batch_size
        self._model = model

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from nemo.collections.asr.models import EncDecMultiTaskModel
        except ImportError as exc:
            raise LoadError(
                "hf: canary_nemo requires the 'nemo-toolkit[asr]' package, "
                f"which is not installed: {exc}"
            ) from exc

        try:
            device = resolve_device(self._device_opt)
            model = EncDecMultiTaskModel.from_pretrained(self.repo_id)
            decode_cfg = model.cfg.decoding
            decode_cfg.beam.beam_size = self.beam_size
            model.change_decoding_strategy(decode_cfg)
            model.eval()
            if device != "cpu":
                model = model.to(device)
        except Exception as exc:
            raise LoadError(f"hf: failed to load canary model {self.repo_id!r}: {exc}") from exc
        self._model = model
        return model

    async def invoke(self, request: Request) -> Response:
        start = time.perf_counter()
        model = self._ensure_loaded()  # LoadError propagates: a setup/config failure

        audio_part = next(
            (p for m in request.messages for p in m.parts if isinstance(p, AudioPart)), None
        )
        if audio_part is None:
            return Response(
                error="canary_nemo: request has no AudioPart",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        if not audio_part.path:
            return Response(
                error="canary_nemo: AudioPart.path is required (NeMo transcribes from file paths)",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        try:
            hypotheses = model.transcribe(
                audio=[audio_part.path],
                batch_size=self.batch_size,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                pnc=self.pnc,
                timestamps=self.timestamps,
            )
            # NeMo returns either list[Hypothesis] or (best_hyps, all_hyps).
            hyp = hypotheses[0] if isinstance(hypotheses, (list, tuple)) else hypotheses
            if isinstance(hyp, (list, tuple)):
                hyp = hyp[0]
            text = getattr(hyp, "text", None)
            if text is None:
                text = str(hyp)
            return Response(text=text, latency_ms=(time.perf_counter() - start) * 1000)
        except Exception as exc:
            return Response(
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    async def aclose(self) -> None:
        self._model = None


def load(repo_id: str, **opts: Any) -> CanaryNemoSystem:
    return CanaryNemoSystem(repo_id, **opts)
