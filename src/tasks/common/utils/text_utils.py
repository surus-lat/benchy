"""Text processing utilities for task handlers."""

import re
import unicodedata
from typing import Optional


def normalize_spaces(text: str) -> str:
    """
    Collapse consecutive whitespace into single spaces and trim leading and trailing whitespace.
    
    Parameters:
        text (str): Input string; if falsy, it is treated as an empty string.
    
    Returns:
        str: The input with runs of whitespace replaced by a single space and surrounding whitespace removed.
    """
    return re.sub(r"\s+", " ", text or "").strip()


def remove_accents(text: str) -> str:
    """
    Remove diacritical marks (accents) from text to simplify matching.
    
    Returns:
        str: The input text with combining diacritical marks removed. Returns an empty string for falsy input.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_text(text: str) -> str:
    """Normalize text for matching (remove accents, lowercase, normalize spaces)."""
    return remove_accents(normalize_spaces(text)).lower()


def extract_float_score(
    response: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    fallback: Optional[float] = None,
) -> Optional[float]:
    """
    Extracts a float value from the given text and optionally clips it to an inclusive range.
    
    Searches the text for numeric patterns (supports both '.' and ',' as decimal separators) and uses the last match. If no parseable number is found or parsing fails, returns `fallback`. Clipping is applied only when both `min_value` and `max_value` are provided.
    
    Parameters:
        min_value (Optional[float]): Minimum allowed value; used for clipping when `max_value` is also provided.
        max_value (Optional[float]): Maximum allowed value; used for clipping when `min_value` is also provided.
        fallback (Optional[float]): Value to return if no numeric value can be extracted or parsed.
    
    Returns:
        Extracted float if a number is found and parsed successfully; `fallback` otherwise.
    """
    if not response:
        return fallback
    
    # Find all numeric patterns (supports both . and , as decimal separator)
    matches = re.findall(r"[-+]?\d+(?:[.,]\d+)?", response)
    if not matches:
        return fallback
    
    # Use the last match (often the final answer)
    raw = matches[-1]
    try:
        value = float(raw.replace(",", "."))
    except ValueError:
        return fallback
    
    # Clip to range if specified
    if min_value is not None and max_value is not None:
        if value < min_value or value > max_value:
            value = max(min(value, max_value), min_value)
    
    return value


def format_score_with_comma(value: float, decimals: int = 1) -> str:
    """
    Format a float using a comma as the decimal separator.
    
    Parameters:
        value: Float value to format.
        decimals: Number of decimal places.
    
    Returns:
        Formatted string with a comma as the decimal separator.
    
    Examples:
        >>> format_score_with_comma(3.5)
        '3,5'
        >>> format_score_with_comma(4.123, decimals=2)
        '4,12'
    """
    return f"{value:.{decimals}f}".replace(".", ",")
