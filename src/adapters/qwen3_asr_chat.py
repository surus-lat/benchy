"""qwen3_asr_chat adapter — Qwen3-ASR family ASR via the `qwen-asr` package.

Qwen3-ASR ships its own runtime: the `qwen-asr` PyPI package, NOT
`transformers` directly. The HF repo declares
`Qwen3ASRForConditionalGeneration` as the architecture, which doesn't
exist in any `transformers` release. `qwen_asr.Qwen3ASRModel` wraps
the model with a dedicated `.transcribe(audio, language=...)` method.

The adapter name keeps the `_chat` suffix for YAML stability — the
field `adapter: qwen3_asr_chat` is part of the public surface — but
the implementation no longer uses a chat template.

**Version conflict note:** `qwen-asr` 0.0.6 pins `transformers <5.0`
while Voxtral needs `transformers >=5.13`. The two can't be active
in the same Python environment. Pick the one you need for the run,
or maintain two venvs.

Config knobs (read from `qwen3_asr_chat:` block in the model YAML):
  dtype: "bfloat16" | "float16" | "float32" (default "bfloat16")
  device_map: str (default "auto") — passed to qwen_asr.from_pretrained
  language: str | None (default None — auto-detect; can force "Spanish")
  max_inference_batch_size: int (default 32)
  max_new_tokens: int (default 256)
  return_time_stamps: bool (default False)
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


# ISO 639-1 → Qwen3-ASR's human-readable language strings. None = auto-detect.
_LANG_NAMES = {
    "es": "Spanish",
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "th": "Thai",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "hi": "Hindi",
    "id": "Indonesian",
    "nl": "Dutch",
    "pl": "Polish",
    "cs": "Czech",
    "ro": "Romanian",
    "el": "Greek",
    "hu": "Hungarian",
    "fa": "Persian",
    "ms": "Malay",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "fil": "Filipino",
    "yue": "Cantonese",
    "mk": "Macedonian",
}


class Adapter(BaseAdapter):
    capabilities = InterfaceCapabilities(supports_audio=True)

    def __init__(self, model_name: str, config: dict[str, Any]):
        super().__init__(model_name, config)
        self._model = None

    def _load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        dtype_name = _DTYPE_MAP[self.config.get("dtype", "bfloat16")]
        torch_dtype = getattr(torch, dtype_name)
        device_map = self.config.get("device_map", "auto")
        max_batch = int(self.config.get("max_inference_batch_size", 32))
        max_new = int(self.config.get("max_new_tokens", 256))

        self._model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            dtype=torch_dtype,
            device_map=device_map,
            max_inference_batch_size=max_batch,
            max_new_tokens=max_new,
        )

    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self._model is None:
            self._load()

        config_lang = self.config.get("language")
        return_ts = bool(self.config.get("return_time_stamps", False))

        results = []
        for req in requests:
            try:
                lang = config_lang
                if lang is None:
                    iso = req.get("language")
                    if iso:
                        lang = _LANG_NAMES.get(iso, None)
                transcribe_result = self._model.transcribe(
                    audio=req["audio_path"],
                    language=lang,
                    return_time_stamps=return_ts,
                )
                hyp = transcribe_result[0]
                text = getattr(hyp, "text", str(hyp))
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
