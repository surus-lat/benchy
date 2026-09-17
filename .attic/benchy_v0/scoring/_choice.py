"""Private choice-parsing helpers for `multiple_choice_accuracy`.

Ported from the old tree's `src/tasks/common/utils/choice_utils.py`
(`parse_choice_prediction` / `parse_choice_index` / `extract_answer_segment`),
trimmed to what the primitive needs. Not part of the public API.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ._text import CHOICE_LABELS, extract_choice_label

__all__ = ["parse_choice_prediction"]

DEFAULT_ANSWER_MARKERS = ("answer", "respuesta", "label", "etiqueta", "salida", "output")


def _normalize_for_matching(text: str) -> str:
    """Fold accents, lowercase, and collapse everything non-alphanumeric to
    spaces, so choice text can be matched loosely inside freeform prose."""
    import unicodedata

    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    lowered = stripped.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def extract_answer_segment(response: str, markers: Sequence[str] = DEFAULT_ANSWER_MARKERS) -> str:
    """Trim `response` down to whatever follows the last answer marker
    (e.g. "Answer:"), to reduce false positives from reasoning text that
    happens to mention another choice before the real answer."""
    lowered = response.lower()
    last_pos = -1
    for marker in markers:
        idx = lowered.rfind(marker)
        if idx > last_pos:
            last_pos = idx
    if last_pos == -1:
        return response.strip()

    segment = response[last_pos:]
    split_idx = segment.find(":")
    if split_idx != -1:
        segment = segment[split_idx + 1 :]
    return segment.strip()


def parse_choice_index(
    prediction: object,
    choices: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    label_to_index: dict | None = None,
    strict: bool = True,
) -> int | None:
    """Parse a 0-based choice index out of a prediction: numeric index,
    numeric label (via `label_to_index`), letter label, or choice text."""
    if prediction is None:
        return None

    if isinstance(prediction, bool):
        prediction = int(prediction)

    if isinstance(prediction, int):
        if 0 <= prediction < len(choices):
            return prediction
        if label_to_index and prediction in label_to_index:
            return label_to_index[prediction]
        return None

    if isinstance(prediction, float) and prediction.is_integer():
        return parse_choice_index(
            int(prediction), choices, labels=labels, label_to_index=label_to_index, strict=strict
        )

    text = str(prediction).strip()
    if not text:
        return None

    if text.lstrip("-").isdigit():
        numeric = int(text)
        if 0 <= numeric < len(choices):
            return numeric
        if label_to_index and numeric in label_to_index:
            return label_to_index[numeric]

    label_list = list(labels) if labels else list(CHOICE_LABELS[: len(choices)])
    letter = extract_choice_label(text, label_list)
    if letter and letter in label_list:
        return label_list.index(letter)

    normalized_response = _normalize_for_matching(text)
    for idx, choice in enumerate(choices):
        choice_norm = _normalize_for_matching(str(choice))
        if choice_norm and choice_norm in normalized_response:
            return idx

    if not strict:
        text_lower = text.lower()
        for idx, choice in enumerate(choices):
            choice_lower = str(choice).lower().strip()
            if len(choice_lower) > 10 and choice_lower[:10] in text_lower:
                return idx

    return None


def parse_choice_prediction(
    prediction: object,
    choices: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    label_to_index: dict | None = None,
    answer_markers: Sequence[str] = DEFAULT_ANSWER_MARKERS,
    strict: bool = True,
) -> int | None:
    """Parse a model prediction (dict, list, JSON string, letter, index, or
    freeform text) into a 0-based choice index, or None if it can't be
    resolved."""
    if prediction is None:
        return None

    if isinstance(prediction, dict):
        for key in ("label", "answer", "prediction", "category", "class"):
            if key in prediction:
                return parse_choice_prediction(
                    prediction[key], choices, labels=labels, label_to_index=label_to_index,
                    answer_markers=answer_markers, strict=strict,
                )
        if len(prediction) == 1:
            return parse_choice_prediction(
                next(iter(prediction.values())), choices, labels=labels,
                label_to_index=label_to_index, answer_markers=answer_markers, strict=strict,
            )
        return None

    if isinstance(prediction, list):
        if len(prediction) == 1:
            return parse_choice_prediction(
                prediction[0], choices, labels=labels, label_to_index=label_to_index,
                answer_markers=answer_markers, strict=strict,
            )
        return None

    if isinstance(prediction, (bool, int, float)):
        return parse_choice_index(prediction, choices, labels=labels, label_to_index=label_to_index, strict=strict)

    text = str(prediction).strip()
    if not text:
        return None

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            return parse_choice_prediction(
                parsed, choices, labels=labels, label_to_index=label_to_index,
                answer_markers=answer_markers, strict=strict,
            )

    answer_text = extract_answer_segment(text, markers=answer_markers)
    return parse_choice_index(answer_text, choices, labels=labels, label_to_index=label_to_index, strict=strict)
