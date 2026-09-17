"""Tolerant choice/label parsing for classification tasks.

Adapted from the salvaged ``src/tasks/common/utils/choice_utils.py`` --
those heuristics (letter markers, "Answer: X" segments, JSON-wrapped
answers, accent/case-insensitive substring matching) were battle-tested
there. The Handler god-object around them was not kept; this is just the
parsing function, made recursive-safe and exception-free.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from typing import Any

DEFAULT_ANSWER_MARKERS: tuple[str, ...] = (
    "answer", "respuesta", "label", "etiqueta", "salida", "output",
)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", stripped.lower()).strip()


def _letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def _extract_letter(text: str, letters: Sequence[str]) -> str | None:
    text = text.strip()
    if not text:
        return None
    for letter in letters:
        if re.search(rf"(?i)(?:^|\b){re.escape(letter)}(?:\b|[).:])", text):
            return letter
    return None


def _answer_segment(text: str, markers: Sequence[str] = DEFAULT_ANSWER_MARKERS) -> str:
    lowered = text.lower()
    last = max((lowered.rfind(marker) for marker in markers), default=-1)
    if last == -1:
        return text.strip()
    segment = text[last:]
    colon = segment.find(":")
    return (segment[colon + 1 :] if colon != -1 else segment).strip()


def parse_label(prediction: Any, labels: Sequence[str]) -> tuple[str | None, str | None]:
    """Resolve a raw model prediction to one of `labels`.

    Returns ``(label, error)`` -- error is None on success. Handles
    letters (A/B/...), 0-based indices, exact/substring text matches
    (accent- and case-insensitive), and JSON-wrapped answers
    (``{"label": "B"}``, ``["B"]``). Never raises.
    """
    if not labels:
        return None, "no labels configured"
    if prediction is None:
        return None, "empty prediction"

    letters = _letters(len(labels))

    if isinstance(prediction, bool):
        prediction = int(prediction)
    if isinstance(prediction, int):
        if 0 <= prediction < len(labels):
            return labels[prediction], None
        return None, f"index {prediction} out of range for {len(labels)} labels"
    if isinstance(prediction, float) and prediction.is_integer():
        return parse_label(int(prediction), labels)
    if isinstance(prediction, (list, tuple)):
        if len(prediction) == 1:
            return parse_label(prediction[0], labels)
        return None, f"expected a single label, got a sequence of {len(prediction)}"
    if isinstance(prediction, dict):
        for key in ("label", "answer", "prediction", "category", "class"):
            if key in prediction:
                return parse_label(prediction[key], labels)
        if len(prediction) == 1:
            return parse_label(next(iter(prediction.values())), labels)
        return None, f"could not find a label key in {prediction!r}"

    text = str(prediction).strip()
    if not text:
        return None, "empty prediction"

    if text.lstrip("-").isdigit():
        idx = int(text)
        if 0 <= idx < len(labels):
            return labels[idx], None

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
        except Exception:  # noqa: BLE001 - fall through to text heuristics
            parsed = None
        if parsed is not None:
            return parse_label(parsed, labels)

    letter = _extract_letter(text, letters)
    if letter:
        return labels[letters.index(letter)], None

    normalized = _normalize(text)
    for label in labels:
        norm_label = _normalize(str(label))
        if norm_label and norm_label in normalized:
            return label, None

    answer_text = _answer_segment(text)
    if answer_text and answer_text != text:
        return parse_label(answer_text, labels)

    return None, f"could not resolve a label from {prediction!r} among {list(labels)!r}"
