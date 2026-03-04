"""Shared helpers for multiple-choice prompts and parsing."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Optional, Sequence

CHOICE_LABELS = [chr(ord("A") + i) for i in range(26)]
DEFAULT_ANSWER_MARKERS = ("answer", "respuesta", "label", "etiqueta", "salida", "output")


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
    strict: bool = True,
) -> Optional[int]:
    """Parse a choice index from a model prediction.

    Supports numeric indices, numeric label values (via label_to_index),
    labeled letters (A/B), or matching against choice text.
    
    Args:
        prediction: Model output (can be int, str, etc.)
        choices: List of choice strings
        labels: Optional choice labels (e.g., ['A', 'B', 'C'])
        label_to_index: Optional mapping of numeric labels to indices
        strict: If False, allows partial matching (first 10 chars). Default True.
    
    Returns:
        Choice index (0-based) or None if not found
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

    # Partial matching for non-strict mode (useful for long choices)
    if not strict:
        text_lower = text.lower()
        for idx, choice in enumerate(choices):
            choice_lower = str(choice).lower().strip()
            # Match first 10 chars if choice is long enough
            if len(choice_lower) > 10 and choice_lower[:10] in text_lower:
                return idx

    return None


def extract_answer_segment(response: str, markers: Sequence[str] = DEFAULT_ANSWER_MARKERS) -> str:
    """Extract the answer segment from a response string.

    Uses the last matching marker (e.g., "Answer:") to reduce false positives.
    """
    # Find the last marker to avoid capturing earlier instruction text.
    lowered = response.lower()
    last_pos = -1
    for marker in markers:
        idx = lowered.rfind(marker)
        if idx > last_pos:
            last_pos = idx
    if last_pos == -1:
        return response.strip()

    # Trim the marker prefix and any colon separator.
    segment = response[last_pos:]
    split_idx = segment.find(":")
    if split_idx != -1:
        segment = segment[split_idx + 1 :]
    return segment.strip()


def parse_choice_prediction(
    prediction: object,
    choices: Sequence[str],
    *,
    labels: Optional[Sequence[str]] = None,
    label_to_index: Optional[dict] = None,
    answer_markers: Sequence[str] = DEFAULT_ANSWER_MARKERS,
    strict: bool = True,
) -> Optional[int]:
    """Parse a model prediction into a choice index.

    Supports numeric labels, letter labels, JSON payloads, or freeform text.
    
    Args:
        prediction: Model output (dict, list, str, int, etc.)
        choices: List of choice strings
        labels: Optional choice labels
        label_to_index: Optional mapping of numeric labels to indices
        answer_markers: Keywords to extract answer segment from response
        strict: If False, allows partial matching. Default True.
    
    Returns:
        Choice index (0-based) or None if not found
    """
    # Fast-path empty predictions.
    if prediction is None:
        return None

    # Parse common JSON-like outputs (dict/list/JSON string).
    if isinstance(prediction, dict):
        for key in ("label", "answer", "prediction", "category", "class"):
            if key in prediction:
                return parse_choice_prediction(
                    prediction[key],
                    choices,
                    labels=labels,
                    label_to_index=label_to_index,
                    answer_markers=answer_markers,
                    strict=strict,
                )
        if len(prediction) == 1:
            return parse_choice_prediction(
                next(iter(prediction.values())),
                choices,
                labels=labels,
                label_to_index=label_to_index,
                answer_markers=answer_markers,
                strict=strict,
            )
        return None

    if isinstance(prediction, list):
        if len(prediction) == 1:
            return parse_choice_prediction(
                prediction[0],
                choices,
                labels=labels,
                label_to_index=label_to_index,
                answer_markers=answer_markers,
                strict=strict,
            )
        return None

    # Handle numeric labels directly.
    if isinstance(prediction, (bool, int, float)):
        return parse_choice_index(
            prediction,
            choices,
            labels=labels,
            label_to_index=label_to_index,
            strict=strict,
        )

    # Fall back to text parsing.
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
                parsed,
                choices,
                labels=labels,
                label_to_index=label_to_index,
                answer_markers=answer_markers,
                strict=strict,
            )

    # Trim to an answer-only segment before parsing.
    answer_text = extract_answer_segment(text, markers=answer_markers)
    return parse_choice_index(
        answer_text,
        choices,
        labels=labels,
        label_to_index=label_to_index,
        strict=strict,
    )
