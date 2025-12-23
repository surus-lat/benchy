"""Metrics calculator for Portuguese tasks."""

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import utils

logger = logging.getLogger(__name__)


class PortugueseMetricsCalculator:
    """Calculator for Portuguese task metrics."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def calculate(
        self,
        prediction: Optional[Any],
        expected: Any,
        sample: Dict[str, Any],
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        task_type: str = "multiple_choice",
        labels: Optional[Sequence[str]] = None,
        score_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for a single sample."""
        if error or prediction is None:
            return self._error_metrics(error or "No prediction", error_type, task_type)

        if task_type == "multiple_choice":
            return self._calculate_multiple_choice(prediction, expected, sample, error_type)
        if task_type == "classification":
            return self._calculate_classification(prediction, expected, labels, error_type)
        if task_type == "regression":
            return self._calculate_regression(prediction, expected, score_range, error_type)

        return self._error_metrics("Unsupported task_type", "invalid_response", task_type)

    def aggregate(
        self,
        all_metrics: List[Dict[str, Any]],
        task_type: str = "multiple_choice",
        labels: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Aggregate per-sample metrics."""
        if not all_metrics:
            return {
                "total_samples": 0,
                "valid_samples": 0,
            }

        valid_metrics = [m for m in all_metrics if m.get("valid", False)]
        total_samples = len(all_metrics)
        valid_samples = len(valid_metrics)

        aggregated: Dict[str, Any] = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "error_rate": (total_samples - valid_samples) / total_samples if total_samples else 0.0,
        }

        if valid_samples == 0:
            if task_type in ("multiple_choice", "classification"):
                aggregated["acc"] = 0.0
            if task_type == "classification":
                aggregated["f1_macro"] = 0.0
            if task_type == "regression":
                aggregated["mse"] = 0.0
                aggregated["pearson"] = 0.0
            return aggregated

        if task_type in ("multiple_choice", "classification"):
            acc_scores = [float(m.get("acc", 0.0)) for m in valid_metrics if "acc" in m]
            aggregated["acc"] = sum(acc_scores) / len(acc_scores) if acc_scores else 0.0

        if task_type == "classification":
            expected_idx = [m.get("expected_idx") for m in valid_metrics]
            predicted_idx = [m.get("predicted_idx") for m in valid_metrics]
            class_count = len(labels) if labels else 2
            aggregated["f1_macro"] = self._f1_macro(predicted_idx, expected_idx, class_count)

        if task_type == "regression":
            mse_scores = [float(m.get("mse", 0.0)) for m in valid_metrics if "mse" in m]
            aggregated["mse"] = sum(mse_scores) / len(mse_scores) if mse_scores else 0.0

            preds = [m.get("prediction") for m in valid_metrics if "prediction" in m]
            refs = [m.get("expected") for m in valid_metrics if "expected" in m]
            aggregated["pearson"] = self._pearsonr(preds, refs)

        return aggregated

    def _error_metrics(self, error: str, error_type: Optional[str], task_type: str) -> Dict[str, Any]:
        base = {
            "valid": False,
            "error": error,
            "error_type": error_type,
        }
        if task_type in ("multiple_choice", "classification"):
            base["acc"] = 0.0
        if task_type == "classification":
            base["predicted_idx"] = None
            base["expected_idx"] = None
        if task_type == "regression":
            base["mse"] = 0.0
            base["prediction"] = None
            base["expected"] = None
        return base

    def _calculate_multiple_choice(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict[str, Any],
        error_type: Optional[str],
    ) -> Dict[str, Any]:
        choices = sample.get("choices", [])
        labels = sample.get("choice_labels") or utils.CHOICE_LABELS[: len(choices)]

        if not choices:
            return self._error_metrics("No choices provided", "invalid_response", "multiple_choice")

        if isinstance(prediction, int):
            selected_idx = prediction
        else:
            letter = utils.extract_choice_letter(str(prediction), labels)
            selected_idx = utils.choice_letter_to_index(letter, labels) if letter else None

        if selected_idx is None or selected_idx < 0 or selected_idx >= len(choices):
            return {
                "valid": False,
                "error": "Could not parse choice from response",
                "error_type": error_type or "invalid_response",
                "acc": 0.0,
            }

        if isinstance(expected, int):
            expected_idx = expected
        elif isinstance(expected, str):
            expected_idx = utils.choice_letter_to_index(expected, labels)
        else:
            expected_idx = None

        if expected_idx is None or expected_idx < 0 or expected_idx >= len(choices):
            return {
                "valid": False,
                "error": "Invalid expected choice",
                "error_type": "invalid_response",
                "acc": 0.0,
            }

        return {
            "valid": True,
            "acc": 1.0 if selected_idx == expected_idx else 0.0,
            "predicted_idx": selected_idx,
            "expected_idx": expected_idx,
        }

    def _calculate_classification(
        self,
        prediction: Any,
        expected: Any,
        labels: Optional[Sequence[str]],
        error_type: Optional[str],
    ) -> Dict[str, Any]:
        label_list = list(labels) if labels else ["Não", "Sim"]

        if isinstance(prediction, int):
            predicted_idx = prediction
        else:
            predicted_label = utils.extract_yes_no_label(str(prediction))
            predicted_idx = label_list.index(predicted_label) if predicted_label in label_list else None

        if predicted_idx is None or predicted_idx < 0 or predicted_idx >= len(label_list):
            return {
                "valid": False,
                "error": "Could not parse label from response",
                "error_type": error_type or "invalid_response",
                "acc": 0.0,
            }

        if isinstance(expected, int):
            expected_idx = expected
        elif isinstance(expected, str):
            expected_idx = label_list.index(expected) if expected in label_list else None
        else:
            expected_idx = None

        if expected_idx is None or expected_idx < 0 or expected_idx >= len(label_list):
            return {
                "valid": False,
                "error": "Invalid expected label",
                "error_type": "invalid_response",
                "acc": 0.0,
            }

        return {
            "valid": True,
            "acc": 1.0 if predicted_idx == expected_idx else 0.0,
            "predicted_idx": predicted_idx,
            "expected_idx": expected_idx,
        }

    def _calculate_regression(
        self,
        prediction: Any,
        expected: Any,
        score_range: Optional[Tuple[float, float]],
        error_type: Optional[str],
    ) -> Dict[str, Any]:
        min_value = score_range[0] if score_range else None
        max_value = score_range[1] if score_range else None
        fallback = None
        if score_range:
            fallback = max_value

        pred_value = utils.extract_float_score(
            str(prediction),
            min_value=min_value,
            max_value=max_value,
            fallback=fallback,
        )

        if pred_value is None:
            return {
                "valid": False,
                "error": "Could not parse score from response",
                "error_type": error_type or "invalid_response",
                "mse": 0.0,
            }

        try:
            expected_value = float(expected)
        except (TypeError, ValueError):
            return {
                "valid": False,
                "error": "Invalid expected score",
                "error_type": "invalid_response",
                "mse": 0.0,
            }

        mse = (pred_value - expected_value) ** 2
        return {
            "valid": True,
            "mse": mse,
            "prediction": pred_value,
            "expected": expected_value,
        }

    def _f1_macro(self, preds: List[Optional[int]], refs: List[Optional[int]], class_count: int) -> float:
        if not preds or not refs or len(preds) != len(refs):
            return 0.0

        tp = [0] * class_count
        fp = [0] * class_count
        fn = [0] * class_count

        for pred, ref in zip(preds, refs):
            if pred is None or ref is None:
                continue
            if pred == ref:
                tp[ref] += 1
            else:
                fp[pred] += 1
                fn[ref] += 1

        f1_scores = []
        for idx in range(class_count):
            precision = tp[idx] / (tp[idx] + fp[idx]) if (tp[idx] + fp[idx]) > 0 else 0.0
            recall = tp[idx] / (tp[idx] + fn[idx]) if (tp[idx] + fn[idx]) > 0 else 0.0
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))

        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    def _pearsonr(self, xs: List[Any], ys: List[Any]) -> float:
        if not xs or not ys or len(xs) != len(ys):
            return 0.0

        try:
            x_vals = [float(x) for x in xs]
            y_vals = [float(y) for y in ys]
        except (TypeError, ValueError):
            return 0.0

        if len(x_vals) < 2:
            return 0.0

        mean_x = sum(x_vals) / len(x_vals)
        mean_y = sum(y_vals) / len(y_vals)
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
        denom_x = sum((x - mean_x) ** 2 for x in x_vals)
        denom_y = sum((y - mean_y) ** 2 for y in y_vals)
        denom = math.sqrt(denom_x * denom_y)

        if denom == 0:
            return 0.0

        return num / denom
