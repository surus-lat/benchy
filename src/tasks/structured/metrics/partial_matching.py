"""Partial matching utilities for field-level comparison."""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class PartialMatcher:
    """Handles partial matching for different data types."""

    def __init__(self, config: Dict):
        """Initialize partial matcher.

        Args:
            config: Configuration dictionary
        """
        self.config = config.get("metrics", {}).get("partial_matching", {})
        self.string_config = self.config.get("string", {})
        self.number_config = self.config.get("number", {})
        self.normalization = config.get("metrics", {}).get("normalization", {})

    def compare_values(self, predicted: Any, expected: Any) -> Tuple[str, float]:
        """Compare two values and return match type and score.

        Args:
            predicted: Predicted value
            expected: Expected value

        Returns:
            Tuple of (match_type, score) where:
            - match_type: 'exact', 'partial', or 'incorrect'
            - score: Float from 0.0 to 1.0
        """
        # Handle None/null
        if predicted is None and expected is None:
            return ('exact', 1.0)
        if predicted is None or expected is None:
            return ('incorrect', 0.0)

        # Handle strings
        if isinstance(expected, str) and isinstance(predicted, str):
            return self._compare_strings(predicted, expected)

        # Handle numbers
        if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
            return self._compare_numbers(predicted, expected)

        # Handle booleans
        if isinstance(expected, bool) and isinstance(predicted, bool):
            score = 1.0 if predicted == expected else 0.0
            match_type = 'exact' if score == 1.0 else 'incorrect'
            return (match_type, score)

        # Handle arrays
        if isinstance(expected, list) and isinstance(predicted, list):
            return self._compare_arrays(predicted, expected)

        # Exact match for other types
        score = 1.0 if predicted == expected else 0.0
        match_type = 'exact' if score == 1.0 else 'incorrect'
        return (match_type, score)

    def _compare_strings(self, predicted: str, expected: str) -> Tuple[str, float]:
        """Compare two strings with composite scoring.

        Args:
            predicted: Predicted string
            expected: Expected string

        Returns:
            Tuple of (match_type, composite_score)
        """
        # Normalize
        pred_norm = self._normalize_string(predicted)
        exp_norm = self._normalize_string(expected)

        # Quick exact check
        if pred_norm == exp_norm:
            return ('exact', 1.0)

        # Calculate composite score
        score = self._composite_score(pred_norm, exp_norm)

        # Classify
        exact_threshold = float(self.string_config.get("exact_threshold", 0.95))
        partial_threshold = float(self.string_config.get("partial_threshold", 0.50))

        if score >= exact_threshold:
            return ('exact', score)
        elif score >= partial_threshold:
            return ('partial', score)
        else:
            return ('incorrect', score)

    def _composite_score(self, predicted: str, expected: str) -> float:
        """Calculate composite similarity score for strings.

        Args:
            predicted: Predicted string (normalized)
            expected: Expected string (normalized)

        Returns:
            Composite score from 0.0 to 1.0
        """
        # 1. Token Overlap F1
        pred_tokens = set(predicted.split())
        exp_tokens = set(expected.split())
        intersection = pred_tokens & exp_tokens

        if pred_tokens and exp_tokens:
            token_p = len(intersection) / len(pred_tokens)
            token_r = len(intersection) / len(exp_tokens)
            token_f1 = 2 * token_p * token_r / (token_p + token_r) if (token_p + token_r) > 0 else 0.0
        else:
            token_f1 = 0.0

        # 2. Levenshtein Similarity
        try:
            from Levenshtein import distance
            max_len = max(len(predicted), len(expected))
            lev_sim = 1 - (distance(predicted, expected) / max_len) if max_len > 0 else 0.0
        except ImportError:
            # Fallback if python-Levenshtein not available
            lev_sim = 1.0 if predicted == expected else 0.0

        # 3. Containment
        if expected in predicted:
            containment = 1.0
        elif predicted in expected:
            containment = len(predicted) / len(expected) if len(expected) > 0 else 0.0
        else:
            containment = 0.0

        # Weighted composite
        token_weight = float(self.string_config.get("token_overlap_weight", 0.5))
        lev_weight = float(self.string_config.get("levenshtein_weight", 0.3))
        cont_weight = float(self.string_config.get("containment_weight", 0.2))

        composite = (token_weight * token_f1 +
                    lev_weight * lev_sim +
                    cont_weight * containment)

        return composite

    def _compare_numbers(self, predicted: float, expected: float) -> Tuple[str, float]:
        """Compare two numbers.

        Args:
            predicted: Predicted number
            expected: Expected number

        Returns:
            Tuple of (match_type, score)
        """
        rel_tol = float(self.number_config.get("relative_tolerance", 0.001))
        abs_tol = float(self.number_config.get("absolute_tolerance", 1e-6))

        if abs(predicted - expected) <= abs_tol + rel_tol * abs(expected):
            return ('exact', 1.0)
        else:
            return ('incorrect', 0.0)

    def _compare_arrays(self, predicted: List, expected: List) -> Tuple[str, float]:
        """Compare two arrays using Jaccard similarity.

        Args:
            predicted: Predicted array
            expected: Expected array

        Returns:
            Tuple of (match_type, score)
        """
        if not expected:
            score = 1.0 if not predicted else 0.0
            return ('exact' if score == 1.0 else 'incorrect', score)

        # Convert to strings for comparison
        pred_set = set(str(item) for item in predicted)
        exp_set = set(str(item) for item in expected)

        intersection = pred_set & exp_set
        union = pred_set | exp_set

        if union:
            score = len(intersection) / len(union)
        else:
            score = 1.0

        if score == 1.0:
            return ('exact', score)
        elif score >= 0.5:
            return ('partial', score)
        else:
            return ('incorrect', score)

    def _normalize_string(self, s: str) -> str:
        """Normalize a string for comparison.

        Args:
            s: String to normalize

        Returns:
            Normalized string
        """
        if not self.normalization.get("case_sensitive", False):
            s = s.lower()

        if self.normalization.get("normalize_whitespace", True):
            s = " ".join(s.split())

        if self.normalization.get("unicode_normalize", True):
            import unicodedata
            s = unicodedata.normalize('NFKD', s)

        return s

