"""Private text-normalization helpers shared by several primitives.

Not part of the public API (leading underscore, not exported from
`benchy.scoring`). Ported from the old tree's
`src/tasks/common/utils/text_utils.py` and
`src/tasks/common/utils/choice_utils.py`, trimmed to what the primitives in
this package actually need.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

__all__ = ["normalize_text", "remove_accents", "tokenize", "CHOICE_LABELS"]

CHOICE_LABELS: tuple[str, ...] = tuple(chr(ord("A") + i) for i in range(26))


def remove_accents(text: str) -> str:
    """Strip combining diacritics: "Córdoba" -> "Cordoba"."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(
    text: str,
    *,
    case_insensitive: bool = True,
    strip: bool = True,
    fold_accents: bool = False,
) -> str:
    """Normalize a string for comparison. Order matters: whitespace, then case,
    then (optionally) accents, so "  CÓRDOBA  " -> "cordoba" when
    `fold_accents=True`."""
    if strip:
        text = re.sub(r"\s+", " ", text).strip()
    if case_insensitive:
        text = text.lower()
    if fold_accents:
        text = remove_accents(text)
    return text


def tokenize(text: object) -> list[str]:
    """Lowercase whitespace tokens. `None` -> []; lists are flattened."""
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        tokens: list[str] = []
        for entry in text:
            tokens.extend(tokenize(entry))
        return tokens
    return str(text).lower().strip().split()


def extract_choice_label(response: str, labels: Sequence[str]) -> str | None:
    """Find the first label (e.g. "A") that appears as its own token in `response`."""
    response = response.strip()
    if not response:
        return None
    for label in labels:
        if len(label) != 1:
            continue
        pattern = rf"(?i)(?:^|\b){re.escape(label)}(?:\b|[).:])"
        if re.search(pattern, response):
            return label
    return None
