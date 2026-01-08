"""Text processing utilities for task handlers."""

import re
import unicodedata
from typing import Optional


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
    """Normalize text for matching (remove accents, lowercase, normalize spaces)."""
    return remove_accents(normalize_spaces(text)).lower()


def extract_float_score(
    response: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    fallback: Optional[float] = None,
) -> Optional[float]:
    """Extract a float score from text, optionally clipping to range.
    
    Args:
        response: Text containing a numeric value
        min_value: Minimum allowed value (clips if out of range)
        max_value: Maximum allowed value (clips if out of range)
        fallback: Value to return if no number found
    
    Returns:
        Extracted float or fallback value
        
    Examples:
        >>> extract_float_score("Score: 4.5", min_value=1.0, max_value=5.0)
        4.5
        >>> extract_float_score("The answer is 3,7", min_value=1.0, max_value=5.0)
        3.7
        >>> extract_float_score("No number here", fallback=0.0)
        0.0
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
    """Format float with comma as decimal separator (for Portuguese/Spanish).
    
    Args:
        value: Float value to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string with comma separator
        
    Example:
        >>> format_score_with_comma(3.5)
        '3,5'
        >>> format_score_with_comma(4.123, decimals=2)
        '4,12'
    """
    return f"{value:.{decimals}f}".replace(".", ",")

