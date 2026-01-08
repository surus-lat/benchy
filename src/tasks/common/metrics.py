"""Lightweight metric helpers for SimpleTask.

These metrics are intentionally small, dependency-free building blocks for
common evaluation patterns (exact match, token F1, etc.). They are meant to be
combined in SimpleTask.metrics for low-boilerplate task definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from .utils.choice_utils import parse_choice_prediction


class Metric(Protocol):
    """Protocol for SimpleTask metrics.

    Implementations should return per-sample values and provide an
    aggregation step (typically a mean over valid samples).
    """

    name: str

    def per_sample(self, prediction: Any, expected: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Return per-sample metrics as a dict."""
        ...

    def aggregate(self, values: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across samples."""
        ...


@dataclass(frozen=True)
class ScalarMetric:
    """Base class for scalar metrics with mean aggregation.

    Subclasses implement compute(), which should return a float per sample.
    """

    name: str

    def per_sample(self, prediction: Any, expected: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {self.name: self.compute(prediction, expected, sample)}

    def compute(self, prediction: Any, expected: Any, sample: Dict[str, Any]) -> float:
        raise NotImplementedError

    def aggregate(self, values: List[Dict[str, Any]]) -> Dict[str, Any]:
        scores = [entry.get(self.name) for entry in values if isinstance(entry.get(self.name), (int, float))]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return {self.name: mean_score}


@dataclass(frozen=True)
class ExactMatch(ScalarMetric):
    """Exact match between prediction and expected.

    The match can be case-insensitive and optionally trims whitespace.
    """

    name: str = "exact_match"
    case_insensitive: bool = True
    strip: bool = True

    def compute(self, prediction: Any, expected: Any, sample: Dict[str, Any]) -> float:
        if prediction is None or expected is None:
            return 0.0
        pred_text = str(prediction)
        exp_text = str(expected)
        if self.strip:
            pred_text = pred_text.strip()
            exp_text = exp_text.strip()
        if self.case_insensitive:
            pred_text = pred_text.lower()
            exp_text = exp_text.lower()
        return 1.0 if pred_text == exp_text else 0.0


def _normalize_tokens(text: Any) -> List[str]:
    # Normalize predictions/targets into lowercase whitespace tokens.
    if text is None:
        return []
    if isinstance(text, list):
        tokens = []
        for entry in text:
            tokens.extend(_normalize_tokens(entry))
        return tokens
    return str(text).lower().strip().split()


@dataclass(frozen=True)
class F1Score(ScalarMetric):
    """Token-level F1 score (whitespace tokens).

    If expected is a list, returns the max F1 over all references.
    """

    name: str = "f1"

    def compute(self, prediction: Any, expected: Any, sample: Dict[str, Any]) -> float:
        pred_tokens = _normalize_tokens(prediction)
        if isinstance(expected, list):
            return max(self._f1(pred_tokens, _normalize_tokens(option)) for option in expected) if expected else 0.0
        expected_tokens = _normalize_tokens(expected)
        return self._f1(pred_tokens, expected_tokens)

    def _f1(self, pred_tokens: List[str], expected_tokens: List[str]) -> float:
        # Compute precision/recall over token overlap.
        if not pred_tokens or not expected_tokens:
            return 0.0
        pred_set = list(pred_tokens)
        expected_set = list(expected_tokens)
        overlap = 0
        expected_counts = {}
        for token in expected_set:
            expected_counts[token] = expected_counts.get(token, 0) + 1
        for token in pred_set:
            count = expected_counts.get(token, 0)
            if count:
                overlap += 1
                expected_counts[token] = count - 1
        precision = overlap / len(pred_set) if pred_set else 0.0
        recall = overlap / len(expected_set) if expected_set else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class MultipleChoiceAccuracy:
    """Accuracy metric for multiple-choice tasks.

    Returns a per-sample payload with accuracy and validity flags based on
    whether a choice index could be parsed.
    """

    name: str = "accuracy"

    def per_sample(self, prediction: Any, expected: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Parse the prediction into a choice index using sample choices.
        parsed = parse_choice_prediction(
            prediction,
            sample.get("choices", []),
            labels=sample.get("choice_labels"),
            label_to_index=sample.get("label_to_index"),
        )
        if parsed is None:
            return {
                "valid": False,
                "accuracy": 0.0,
                "correct": False,
                "error": "Could not parse label from response",
                "error_type": "invalid_response",
            }

        is_correct = parsed == expected
        return {
            "valid": True,
            "accuracy": 1.0 if is_correct else 0.0,
            "correct": is_correct,
        }

    def aggregate(self, values: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Aggregate accuracy over valid samples only.
        valid = [entry for entry in values if entry.get("valid")]
        accuracy = sum(entry.get("accuracy", 0.0) for entry in valid) / len(valid) if valid else 0.0
        return {"accuracy": accuracy}


@dataclass(frozen=True)
class MeanSquaredError(ScalarMetric):
    """Mean Squared Error for regression tasks.
    
    Computes (prediction - expected)^2 per sample.
    """
    
    name: str = "mse"
    
    def compute(self, prediction: Any, expected: Any, sample: Dict[str, Any]) -> float:
        try:
            pred_val = float(prediction)
            exp_val = float(expected)
            return (pred_val - exp_val) ** 2
        except (TypeError, ValueError):
            return 0.0


@dataclass(frozen=True)
class PearsonCorrelation:
    """Pearson correlation coefficient for regression tasks.
    
    Requires aggregation across all samples to compute correlation.
    """
    
    name: str = "pearson"
    
    def per_sample(self, prediction: Any, expected: Any, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Store individual values for correlation computation during aggregation
        try:
            pred_val = float(prediction)
            exp_val = float(expected)
            return {
                "prediction": pred_val,
                "expected": exp_val,
                "valid": True,
            }
        except (TypeError, ValueError):
            return {"valid": False}
    
    def aggregate(self, values: List[Dict[str, Any]]) -> Dict[str, Any]:
        import math
        
        valid = [entry for entry in values if entry.get("valid")]
        if len(valid) < 2:
            return {"pearson": 0.0}
        
        predictions = [entry["prediction"] for entry in valid]
        expectations = [entry["expected"] for entry in valid]
        
        # Compute Pearson correlation
        mean_pred = sum(predictions) / len(predictions)
        mean_exp = sum(expectations) / len(expectations)
        
        numerator = sum((p - mean_pred) * (e - mean_exp) for p, e in zip(predictions, expectations))
        denom_pred = sum((p - mean_pred) ** 2 for p in predictions)
        denom_exp = sum((e - mean_exp) ** 2 for e in expectations)
        denominator = math.sqrt(denom_pred * denom_exp)
        
        if denominator == 0:
            return {"pearson": 0.0}
        
        return {"pearson": numerator / denominator}
