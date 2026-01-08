"""Shared helpers for multiple-choice prompts and parsing."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Iterable, List, Optional, Sequence

CHOICE_LABELS = [chr(ord("A") + i) for i in range(26)]
DEFAULT_ANSWER_MARKERS = ("answer", "respuesta", "label", "etiqueta", "salida", "output")


def normalize_text(text: str) -> str:
    """
    Normalize a string for loose, case-insensitive matching.
    
    The returned string has diacritics removed, is lowercased, and any non-alphanumeric runs are collapsed to single spaces; leading and trailing whitespace is trimmed.
    
    Parameters:
        text (str): Input text to normalize.
    
    Returns:
        str: The cleaned, lowercase, space-separated string suitable for fuzzy comparisons.
    """
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    lowered = stripped.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def format_choices(choices: Sequence[str], labels: Optional[Sequence[str]] = None) -> str:
    """
    Format a sequence of choices as labeled lines (for example, "A. Choice 1").
    
    Parameters:
        choices (Sequence[str]): The choice texts to format.
        labels (Optional[Sequence[str]]): Optional sequence of labels to use (e.g., ["A", "B", "C"]). If omitted, the default label set (A–Z) is used; if the provided labels are shorter than the number of choices, remaining labels fall back to the default set.
    
    Returns:
        formatted (str): A single string with each labeled choice on its own line, in the form "Label. Choice".
    """
    label_list = list(labels) if labels else CHOICE_LABELS
    lines = []
    for idx, choice in enumerate(choices):
        label = label_list[idx] if idx < len(label_list) else CHOICE_LABELS[idx]
        lines.append(f"{label}. {choice}")
    return "\n".join(lines)


def extract_choice_label(response: str, labels: Sequence[str]) -> Optional[str]:
    """
    Return the first single-character label from `labels` that appears in `response`.
    
    Search is case-insensitive and matches a label at a word boundary or immediately before common punctuation (')', '.', ':'). Labels longer than one character are ignored.
    
    Parameters:
        response (str): The text to search for a label.
        labels (Sequence[str]): Sequence of candidate labels (e.g., ["A", "B", "C"]).
    
    Returns:
        The matching label (as provided in `labels`) if found, `None` otherwise.
    """
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
    """
    Determine the zero-based choice index corresponding to a model prediction.
    
    Parameters:
        prediction: Model output to parse (may be int, float, bool, str, etc.).
        choices (Sequence[str]): Sequence of choice strings to match against.
        labels (Optional[Sequence[str]]): Optional label sequence (e.g., ['A', 'B', 'C']) used for letter-based matching; defaults to A–Z for the number of choices.
        label_to_index (Optional[dict]): Optional mapping from numeric label values to choice indices.
        strict (bool): If False, allow partial matching of long choice texts (first 10 characters). Defaults to True.
    
    Returns:
        Optional[int]: The matched 0-based choice index if found, `None` otherwise.
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
    """
    Extracts the answer portion from a freeform response using known answer markers.
    
    Searches for the last occurrence of any marker (case-insensitive) and returns the text after that marker, trimming surrounding whitespace. If a colon immediately follows the marker it will be removed. If no marker is found, returns the trimmed original response.
    
    Parameters:
        response (str): The raw response text to extract the answer from.
        markers (Sequence[str]): Marker strings to locate the answer segment (case-insensitive). Defaults to DEFAULT_ANSWER_MARKERS.
    
    Returns:
        str: The trimmed answer segment extracted from the response.
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
    """
    Parse a model prediction into a zero-based index for the provided choices.
    
    Attempts multiple parsing strategies: handles dict/list wrappers and common keys, numeric labels (bool/int/float or numeric strings), JSON-encoded payloads, single-letter labels, and freeform text by extracting answer segments and performing label or text matching. Uses strict or permissive matching depending on `strict`.
    
    Parameters:
        prediction (object): Model output to parse (may be dict, list, str, int, float, or bool).
        choices (Sequence[str]): Sequence of candidate choice strings to map against.
        labels (Optional[Sequence[str]]): Optional explicit labels corresponding to `choices` (e.g., ["A","B","C"]).
        label_to_index (Optional[dict]): Optional mapping from numeric or string labels to choice indices.
        answer_markers (Sequence[str]): Markers used to extract the answer portion from freeform text.
        strict (bool): If False, allows partial/looser matching when exact matches fail.
    
    Returns:
        Optional[int]: Zero-based index of the matched choice if found, `None` otherwise.
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