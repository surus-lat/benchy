"""Partial matching utilities for field-level comparison."""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class PartialMatcher:
    """Handles partial matching for different data types.
    
    Supports a 'strict' mode that tightens thresholds for more rigorous evaluation.
    """

    def __init__(self, config: Dict, strict: bool = False):
        """
        Initialize the PartialMatcher with configuration values and an optional strict mode.
        
        Parameters:
            config (Dict): Configuration dictionary. Function reads:
                - metrics.partial_matching (dict): partial matching settings.
                    - string (dict): string thresholds and weights:
                        - exact_threshold (float)
                        - partial_threshold (float)
                        - token_overlap_weight (float)
                        - levenshtein_weight (float)
                        - containment_weight (float)
                    - number (dict): numeric tolerances:
                        - relative_tolerance (float)
                        - absolute_tolerance (float)
                - metrics.normalization (dict): normalization options:
                    - case_sensitive (bool)
                    - normalize_whitespace (bool)
                    - unicode_normalize (bool)
            strict (bool): If True, enable stricter matching thresholds and tolerances.
        """
        self.config = config.get("metrics", {}).get("partial_matching", {})
        self.string_config = self.config.get("string", {})
        self.number_config = self.config.get("number", {})
        self.normalization = config.get("metrics", {}).get("normalization", {})
        self.strict = strict

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
        result = self._compare_values_with_severity(predicted, expected)
        return (result[0], result[1])

    def compare_values_with_severity(self, predicted: Any, expected: Any) -> Tuple[str, float, str]:
        """
        Determine match type, score, and error severity for the given predicted and expected values.
        
        Parameters:
            predicted: The value produced by a model or system.
            expected: The reference value to compare against.
        
        Returns:
            tuple: (match_type, score, error_severity)
                - match_type: 'exact', 'partial', or 'incorrect'
                - score: float between 0.0 and 1.0 indicating similarity
                - error_severity: 'none', 'minor', or 'critical'
        """
        return self._compare_values_with_severity(predicted, expected)

    def _compare_values_with_severity(self, predicted: Any, expected: Any) -> Tuple[str, float, str]:
        """
        Compare two values and classify their match type, numeric score, and error severity.
        
        Parameters:
            predicted (Any): The value produced by the system.
            expected (Any): The reference value to compare against.
        
        Returns:
            Tuple[str, float, str]: A tuple of (match_type, score, error_severity):
                - match_type: 'exact', 'partial', or 'incorrect'.
                - score: similarity score between 0.0 and 1.0.
                - error_severity: 'none', 'minor', or 'critical'.
        
        Behavior notes:
            - Both values None -> ('exact', 1.0, 'none'); one None -> ('incorrect', 0.0, 'critical').
            - Strings are compared for exact, partial, or incorrect similarity; casing/whitespace-only differences are treated as minor.
            - Numeric comparisons yield 'exact' or 'incorrect'; non-exact numeric differences are treated as critical.
            - Boolean mismatches are treated as critical.
            - Values of differing types produce ('incorrect', 0.0, 'critical').
            - Lists are compared for overlap; exact, partial, or incorrect outcomes map to severities none, minor, or critical respectively.
            - For other types, equality yields exact/none, inequality yields incorrect/critical.
        """
        # Handle None/null
        if predicted is None and expected is None:
            return ('exact', 1.0, 'none')
        if predicted is None or expected is None:
            return ('incorrect', 0.0, 'critical')  # Missing values are critical

        # Handle strings
        if isinstance(expected, str) and isinstance(predicted, str):
            match_type, score = self._compare_strings(predicted, expected)
            # Determine severity for string mismatches
            severity = 'none' if match_type == 'exact' else 'minor' if match_type == 'partial' else 'minor'
            # Check if it's just a casing/whitespace issue
            if match_type != 'exact' and self._normalize_string(predicted) == self._normalize_string(expected):
                severity = 'minor'
            elif match_type == 'incorrect':
                severity = 'minor'  # String errors are generally minor unless they're completely wrong
            return (match_type, score, severity)

        # Handle numbers
        if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
            match_type, score = self._compare_numbers(predicted, expected)
            # Numeric errors are critical
            severity = 'none' if match_type == 'exact' else 'critical'
            return (match_type, score, severity)

        # Handle booleans
        if isinstance(expected, bool) and isinstance(predicted, bool):
            score = 1.0 if predicted == expected else 0.0
            match_type = 'exact' if score == 1.0 else 'incorrect'
            # Boolean errors are critical
            severity = 'none' if match_type == 'exact' else 'critical'
            return (match_type, score, severity)

        # Handle type mismatches
        if type(expected) != type(predicted):
            return ('incorrect', 0.0, 'critical')  # Type mismatches are critical

        # Handle arrays
        if isinstance(expected, list) and isinstance(predicted, list):
            match_type, score = self._compare_arrays(predicted, expected)
            severity = 'none' if match_type == 'exact' else 'minor' if match_type == 'partial' else 'critical'
            return (match_type, score, severity)

        # Exact match for other types
        score = 1.0 if predicted == expected else 0.0
        match_type = 'exact' if score == 1.0 else 'incorrect'
        severity = 'none' if match_type == 'exact' else 'critical'
        return (match_type, score, severity)

    def _compare_strings(self, predicted: str, expected: str) -> Tuple[str, float]:
        """
        Classifies similarity between two strings as 'exact', 'partial', or 'incorrect' and returns a similarity score.
        
        Returns:
            (match_type, score): `match_type` is one of `'exact'`, `'partial'`, or `'incorrect'`; `score` is a float in [0.0, 1.0] representing the computed similarity between the normalized `predicted` and `expected` strings.
        """
        # Normalize
        pred_norm = self._normalize_string(predicted)
        exp_norm = self._normalize_string(expected)

        # Quick exact check
        if pred_norm == exp_norm:
            return ('exact', 1.0)

        # Calculate composite score
        score = self._composite_score(pred_norm, exp_norm)

        # Classify with configurable thresholds
        # In strict mode, require higher scores for partial matches
        if self.strict:
            exact_threshold = 0.98  # Stricter: almost perfect match required
            partial_threshold = 0.70  # Stricter: 70% similarity minimum
        else:
            exact_threshold = float(self.string_config.get("exact_threshold", 0.95))
            partial_threshold = float(self.string_config.get("partial_threshold", 0.50))

        if score >= exact_threshold:
            return ('exact', score)
        elif score >= partial_threshold:
            return ('partial', score)
        else:
            return ('incorrect', score)

    def _composite_score(self, predicted: str, expected: str) -> float:
        """
        Compute a weighted similarity score between two normalized strings.
        
        Parameters:
            predicted (str): Predicted string, already normalized.
            expected (str): Expected string, already normalized.
        
        Returns:
            composite (float): Similarity score in the range 0.0 to 1.0, where higher values indicate greater similarity.
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
        """
        Determine whether two numbers match within configured tolerances.
        
        Uses strict-mode tolerances when enabled; otherwise reads relative and absolute tolerances from the instance configuration.
        
        Parameters:
            predicted (float): The predicted numeric value.
            expected (float): The expected numeric value.
        
        Returns:
            Tuple[str, float]: `('exact', 1.0)` if |predicted - expected| <= absolute_tolerance + relative_tolerance * |expected|, `('incorrect', 0.0)` otherwise.
        """
        # In strict mode, use tighter tolerances
        if self.strict:
            rel_tol = 0.0001  # 0.01% relative tolerance
            abs_tol = 1e-9  # Very small absolute tolerance
        else:
            rel_tol = float(self.number_config.get("relative_tolerance", 0.001))
            abs_tol = float(self.number_config.get("absolute_tolerance", 1e-6))

        if abs(predicted - expected) <= abs_tol + rel_tol * abs(expected):
            return ('exact', 1.0)
        else:
            return ('incorrect', 0.0)

    def _compare_arrays(self, predicted: List, expected: List) -> Tuple[str, float]:
        """
        Compute a Jaccard-based similarity between two lists and classify the match as 'exact', 'partial', or 'incorrect'.
        
        Elements of both lists are converted to strings before comparison. If the expected list is empty, the result is 'exact' with score 1.0 only when the predicted list is also empty; otherwise the score is 0.0. The Jaccard score is |intersection| / |union| when the union is non-empty, or 1.0 when both sets are empty.
        
        Parameters:
            predicted (List): Predicted list of elements (elements will be stringified for comparison).
            expected (List): Expected list of elements (elements will be stringified for comparison).
        
        Returns:
            Tuple[str, float]: A tuple of (match_type, score)
                - match_type: 'exact' when score == 1.0, 'partial' when score >= partial_threshold, 'incorrect' otherwise.
                - score: Jaccard similarity between 0.0 and 1.0.
        
        Notes:
            The partial_threshold is 0.70 when the matcher is in strict mode, otherwise 0.50.
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

        # In strict mode, require higher overlap for partial matches
        partial_threshold = 0.70 if self.strict else 0.5

        if score == 1.0:
            return ('exact', score)
        elif score >= partial_threshold:
            return ('partial', score)
        else:
            return ('incorrect', score)

    def _normalize_string(self, s: str) -> str:
        """
        Normalize a string according to the instance's normalization settings.
        
        Parameters:
            s (str): Input string to normalize.
        
        Description:
            Applies the following transformations based on self.normalization:
            - If "case_sensitive" is False, convert to lowercase.
            - If "normalize_whitespace" is True, collapse consecutive whitespace to single spaces and trim leading/trailing whitespace.
            - If "unicode_normalize" is True, apply Unicode NFKD normalization.
        
        Returns:
            str: The normalized string.
        """
        if not self.normalization.get("case_sensitive", False):
            s = s.lower()
        if self.normalization.get("normalize_whitespace", True):
            s = " ".join(s.split())
        if self.normalization.get("unicode_normalize", True):
            import unicodedata
            s = unicodedata.normalize('NFKD', s)
        return s
