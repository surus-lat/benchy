"""Shared helpers for multiple-choice prompts and parsing."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional, Sequence

CHOICE_LABELS = [chr(ord("A") + i) for i in range(26)]


def normalize_text(text: str) -> str:
    """Normalize text for loose matching."""
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    lowered = stripped.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def format_choices(choices: Sequence[str], labels: Optional[Sequence[str]] = None) -> str:
    """Format choices with labels (A., B., ...)."""
    label_list = list(labels) if labels else CHOICE_LABELS
    lines = []
    for idx, choice in enumerate(choices):
        label = label_list[idx] if idx < len(label_list) else CHOICE_LABELS[idx]
        lines.append(f"{label}. {choice}")
    return "\n".join(lines)


def extract_choice_label(response: str, labels: Sequence[str]) -> Optional[str]:
    """Extract a choice label letter from a response string."""
    response_text = response.strip()
    if not response_text:
        return None
    for label in labels:
        if len(label) != 1:
            continue
        pattern = rf"(?i)(?:^|\b){re.escape(label)}(?:\b|[).:])"
        if re.search(pattern, response_text):
            return label
    return None


def parse_choice_index(
    prediction: object,
    choices: Sequence[str],
    *,
    labels: Optional[Sequence[str]] = None,
    label_to_index: Optional[dict] = None,
) -> Optional[int]:
    """Parse a choice index from a model prediction.

    Supports numeric indices, numeric label values (via label_to_index),
    labeled letters (A/B), or matching against choice text.
    """
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
        return parse_choice_index(int(prediction), choices, labels=labels, label_to_index=label_to_index)

    text = str(prediction).strip()
    if not text:
        return None

    if text.lstrip("-").isdigit():
        try:
            numeric = int(text)
        except ValueError:
            numeric = None
        if numeric is not None:
            if 0 <= numeric < len(choices):
                return numeric
            if label_to_index and numeric in label_to_index:
                return label_to_index[numeric]

    label_list = list(labels) if labels else CHOICE_LABELS[: len(choices)]
    letter = extract_choice_label(text, label_list)
    if letter and letter in label_list:
        return label_list.index(letter)

    normalized_response = normalize_text(text)
    for idx, choice in enumerate(choices):
        choice_norm = normalize_text(str(choice))
        if choice_norm and choice_norm in normalized_response:
            return idx

    return None
