"""Transcription handler base class.

Extends ``FreeformHandler`` with transcription-specific behavior:
- WER / CER / ExactMatch metrics
- Per-locale aggregation buckets
- Normalization of predictions/expected text before scoring

Subtasks should implement ``load_dataset()`` (returning a list of sample
dicts with ``id``, ``audio_path``, ``expected``, ``language``, ``locale``)
and set their own ``name``, ``language``, ``locale``, and ``dataset_config``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ..common import FreeformHandler
from ..common.metrics import CharErrorRate, ExactMatch, WordErrorRate

logger = logging.getLogger(__name__)


class TranscriptionHandler(FreeformHandler):
    """Base handler for ASR transcription tasks."""

    answer_type = "freeform"
    requires_audio = True

    # Subtasks override these
    language: str = "es"
    locale: str = "es"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Default metrics for transcription. Subclasses can override by setting
        # ``self.metrics`` in their own __init__ after calling super().
        if not self.metrics or not any(
            getattr(m, "name", None) in ("wer", "cer") for m in self.metrics
        ):
            self.metrics = [WordErrorRate(), CharErrorRate(), ExactMatch()]

    def get_prompt(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        # Whisper-style transcription endpoints take audio, not prompts.
        return ("", "")

    def _normalize_transcription(self, text: Any) -> str:
        if text is None:
            return ""
        return " ".join(str(text).lower().split())

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict[str, Any],
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        if error or prediction is None:
            return {
                "valid": False,
                "error": error,
                "error_type": error_type,
                "locale": sample.get("locale", self.locale),
            }

        pred = self._normalize_transcription(prediction)
        exp = self._normalize_transcription(expected)

        result: Dict[str, Any] = {
            "valid": True,
            "locale": sample.get("locale", self.locale),
        }
        for metric in self.metrics:
            try:
                result.update(metric.per_sample(pred, exp, sample))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Error calculating {metric.__class__.__name__}: {exc}")
        return result

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        if not all_metrics:
            return {
                "total_samples": 0,
                "valid_samples": 0,
                "error_rate": 0.0,
                "per_locale": {},
            }

        valid = [m for m in all_metrics if m.get("valid")]

        # Pre-init so the key always exists even when every sample failed.
        overall: Dict[str, Any] = {metric.name: None for metric in self.metrics}
        for metric in self.metrics:
            entries = [m for m in valid if metric.name in m]
            if entries:
                overall.update(metric.aggregate(entries))

        by_locale: Dict[str, List[Dict]] = defaultdict(list)
        for m in valid:
            by_locale[m.get("locale", self.locale)].append(m)

        per_locale: Dict[str, Dict[str, Any]] = {}
        for loc, samples in by_locale.items():
            locale_metrics: Dict[str, Any] = {}
            for metric in self.metrics:
                entries = [m for m in samples if metric.name in m]
                if entries:
                    locale_metrics.update(metric.aggregate(entries))
            locale_metrics["sample_count"] = len(samples)
            per_locale[loc] = locale_metrics

        overall["per_locale"] = per_locale
        overall["valid_samples"] = len(valid)
        overall["total_samples"] = len(all_metrics)
        overall["error_rate"] = round(
            1.0 - (len(valid) / len(all_metrics)) if all_metrics else 0.0, 4
        )
        return overall
