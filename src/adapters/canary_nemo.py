"""canary_nemo adapter — NVIDIA Canary ASR via the NeMo toolkit.

NVIDIA Canary models (e.g. `nvidia/canary-1b-flash`) are FastConformer
encoder-decoder ASR models that don't load through `transformers` —
the `fastconformer` model_type lives only inside NVIDIA's NeMo
toolkit. This adapter loads them through `nemo.collections.asr.models`
and calls `model.transcribe([audio_paths])` for inference.

Canary-1b-flash supports en/de/es/fr only; pt_br samples will produce
nonsense or fail validation. Use only for supported locales.

Config knobs (read from `canary_nemo:` block in the model YAML):
  device: "auto" | "cpu" | "cuda" | "mps" (default "auto")
  source_lang: str (default "es") — language code of the input audio
  target_lang: str (default same as source_lang) — transcription target
  pnc: "yes" | "no" (default "yes") — punctuation + capitalization
  beam_size: int (default 1) — decoding beam width
  timestamps: bool (default False)
  batch_size: int (default 1)
"""

from __future__ import annotations

from typing import Any

from src.adapters.base import BaseAdapter
from src.engine.protocols import InterfaceCapabilities


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
        self._device = "cpu"

    def _load(self) -> None:
        from nemo.collections.asr.models import EncDecMultiTaskModel

        device = _resolve_device(self.config.get("device", "auto"))
        self._device = device

        self._model = EncDecMultiTaskModel.from_pretrained(self.model_name)
        # Pick beam size before decoding config is frozen.
        decode_cfg = self._model.cfg.decoding
        decode_cfg.beam.beam_size = int(self.config.get("beam_size", 1))
        self._model.change_decoding_strategy(decode_cfg)
        self._model.eval()
        if device != "cpu":
            self._model = self._model.to(device)

    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self._model is None:
            self._load()

        source_lang = self.config.get("source_lang", "es")
        target_lang = self.config.get("target_lang", source_lang)
        pnc = self.config.get("pnc", "yes")
        timestamps = bool(self.config.get("timestamps", False))
        batch_size = int(self.config.get("batch_size", 1))

        results = []
        for req in requests:
            try:
                hypotheses = self._model.transcribe(
                    audio=[req["audio_path"]],
                    batch_size=batch_size,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    pnc=pnc,
                    timestamps=timestamps,
                )
                # NeMo returns either a list[Hypothesis] or a tuple of
                # (best_hyps, all_hyps); take the first hypothesis text.
                hyp = hypotheses[0] if isinstance(hypotheses, (list, tuple)) else hypotheses
                if isinstance(hyp, (list, tuple)):
                    hyp = hyp[0]
                text = getattr(hyp, "text", None)
                if text is None:
                    # Older NeMo versions returned plain strings.
                    text = str(hyp)
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
