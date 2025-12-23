"""Utility helpers for Portuguese tasks."""

import re
import unicodedata
from typing import Iterable, List, Optional, Sequence

CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def normalize_spaces(text: str) -> str:
    """Collapse whitespace and trim."""
    return re.sub(r"\s+", " ", text or "").strip()


def remove_accents(text: str) -> str:
    """Remove accents for simpler matching."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_text(text: str) -> str:
    """Normalize text for label matching."""
    return remove_accents(normalize_spaces(text)).lower()


def format_choices(choices: Sequence[str], labels: Optional[Sequence[str]] = None) -> str:
    """Format choices as labeled lines."""
    labels = list(labels) if labels else CHOICE_LABELS
    lines = []
    for idx, choice in enumerate(choices):
        label = labels[idx] if idx < len(labels) else CHOICE_LABELS[idx]
        lines.append(f"{label}. {choice}")
    return "\n".join(lines)


def _choice_patterns(letter_group: str) -> List[str]:
    return [
        rf"(?:[Ll]etra|[Aa]lternativa|[Rr]esposta|[Rr]esposta [Cc]orreta|[Rr]esposta [Cc]orreta e|[Oo]pcao):? ([{letter_group}])\\b",
        rf"\\b([{letter_group}])\\.",
        rf"\\b([{letter_group}]) ?[.):-]",
        rf"\\b([{letter_group}])$",
        rf"\\b([{letter_group}])\\b",
    ]


def extract_choice_letter(
    response: str,
    labels: Sequence[str],
) -> Optional[str]:
    """Extract a choice letter from a response."""
    if not response:
        return None

    cleaned = remove_accents(normalize_spaces(response))
    label_candidates = [label for label in labels if len(label) == 1]
    if not label_candidates:
        label_candidates = CHOICE_LABELS[: len(labels)]

    letter_group = "".join(label_candidates)
    for pattern in _choice_patterns(letter_group):
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    for letter in label_candidates:
        if re.search(rf"\b{re.escape(letter)}\b", cleaned, flags=re.IGNORECASE):
            return letter.upper()

    return None


def choice_letter_to_index(letter: str, labels: Sequence[str]) -> Optional[int]:
    """Convert a choice letter to its index."""
    if not letter:
        return None
    normalized = letter.upper()
    label_candidates = list(labels) if labels else CHOICE_LABELS
    if normalized in label_candidates:
        return label_candidates.index(normalized)

    fallback = CHOICE_LABELS[: len(label_candidates)]
    if normalized in fallback:
        return fallback.index(normalized)

    return None


def extract_yes_no_label(response: str) -> Optional[str]:
    """Extract a Sim/Não label from the response."""
    if not response:
        return None

    normalized = normalize_text(response)
    matches = []
    for label in ("sim", "nao"):
        match = re.search(rf"\b{label}\b", normalized)
        if match:
            matches.append((match.start(), label))

    if matches:
        matches.sort(key=lambda item: item[0])
        selected = matches[0][1]
    else:
        if "sim" in normalized:
            selected = "sim"
        elif "nao" in normalized:
            selected = "nao"
        else:
            return None

    return "Sim" if selected == "sim" else "Não"


def extract_float_score(
    response: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    fallback: Optional[float] = None,
) -> Optional[float]:
    """Extract a float score from text, optionally clipping to range."""
    if not response:
        return fallback

    matches = re.findall(r"[-+]?\d+(?:[.,]\d+)?", response)
    if not matches:
        return fallback

    raw = matches[-1]
    try:
        value = float(raw.replace(",", "."))
    except ValueError:
        return fallback

    if min_value is not None and max_value is not None:
        if value < min_value or value > max_value:
            value = max(min(value, max_value), min_value)

    return value


def format_score_pt(value: float) -> str:
    """Format float with comma decimal separator (1 decimal place)."""
    return f"{value:.1f}".replace(".", ",")


def build_fewshot_block(examples: Iterable[str]) -> str:
    """Join few-shot examples with blank lines."""
    cleaned = [example.strip() for example in examples if example and example.strip()]
    if not cleaned:
        return ""
    return "\n\n".join(cleaned).strip() + "\n\n"
