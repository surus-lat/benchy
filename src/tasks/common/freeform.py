"""Freeform text format handler.

This handler encapsulates logic for freeform text generation tasks including:
- Exact match scoring
- F1 score calculation
- BLEU score (optional)
- Text normalization
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .metrics import ExactMatch, F1Score
from .base import BaseHandler

logger = logging.getLogger(__name__)


class FreeformHandler(BaseHandler):
    """Handler for freeform text generation tasks.
    
    This handler provides support for open-ended text generation evaluation using
    standard metrics like exact match, F1 score, and optional BLEU score.
    
    Class Attributes (in addition to BaseHandler):
        answer_type: Set to "freeform"
        normalize_prediction: Whether to normalize text for comparison (default: True)
        case_sensitive: Whether comparison is case-sensitive (default: False)
    
    Example:
        class MyTranslationTask(FreeformHandler):
            dataset = "org/translation-dataset"
            system_prompt = "You are a translator."
            metrics = [ExactMatch(), F1Score(), BLEUScore()]
    """

    # Freeform defaults
    answer_type = "freeform"
    normalize_prediction: bool = True
    case_sensitive: bool = False

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FreeformHandler with optional configuration.
        
        Parameters:
            config (Optional[Dict[str, Any]]): Configuration passed to the base handler used to customize behavior (e.g., metrics, text fields). If not provided or if no metrics are specified, default metrics `ExactMatch` and `F1Score` are assigned.
        """
        super().__init__(config)

        # Set default metrics if not provided
        if not self.metrics:
            self.metrics = [ExactMatch(), F1Score()]

    def preprocess_sample(
        self, raw_sample: Dict[str, Any], idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a raw dataset sample into the handler's evaluation sample format.
        
        Parameters:
            raw_sample (Dict[str, Any]): Original dataset entry; expected to contain the handler's configured text_field and label_field.
            idx (int): Index of the sample used to build a unique sample id.
        
        Returns:
            Optional[Dict[str, Any]]: A dict with keys:
                - `id` (str): "{task_name}_{idx}" unique identifier.
                - `text` (str): Stringified input text.
                - `expected` (str): Stringified expected output.
            Returns `None` if either the input text or expected label is missing.
        """
        text = raw_sample.get(self.text_field)
        expected = raw_sample.get(self.label_field)

        if text is None or expected is None:
            logger.warning(f"Skipping sample {idx}: missing text or expected output")
            return None

        return {
            "id": f"{self.get_task_name()}_{idx}",
            "text": str(text),
            "expected": str(expected),
        }

    def get_prompt(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """
        Builds the system and user prompts for a freeform generation sample.
        
        If the handler has a `user_prompt_template` attribute, the user prompt is produced by formatting that template with the sample; otherwise the sample's `"text"` field is used (empty string if missing).
        
        Parameters:
            sample (Dict[str, Any]): Preprocessed sample containing at least a `"text"` key or keys referenced by `user_prompt_template`.
        
        Returns:
            Tuple[str, str]: (system_prompt, user_prompt) where `system_prompt` is taken from the handler's `system_prompt` attribute and `user_prompt` is constructed as described above.
        """
        # Use custom template if provided, otherwise simple format
        if hasattr(self, "user_prompt_template"):
            user_prompt = self.user_prompt_template.format(**sample)
        else:
            user_prompt = sample.get("text", "")

        return self.system_prompt, user_prompt

    def _normalize_text(self, text: str) -> str:
        """
        Normalize a text string for comparison according to the handler's normalization settings.
        
        If `normalize_prediction` is False the input is returned unchanged; otherwise the text is stripped of leading/trailing whitespace, optionally lowercased depending on `case_sensitive`, and internal whitespace is collapsed to single spaces.
        
        Parameters:
            text (str): The input text to normalize.
        
        Returns:
            str: The normalized text.
        """
        if not self.normalize_prediction:
            return text

        # Basic normalization
        normalized = text.strip()

        if not self.case_sensitive:
            normalized = normalized.lower()

        # Normalize whitespace
        normalized = " ".join(normalized.split())

        return normalized

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict[str, Any],
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a prediction against the expected value using the handler's configured metrics.
        
        If `error` is provided, returns the standardized error metrics produced by `get_error_metrics`.
        String predictions and expected values are normalized according to the handler's normalization settings before evaluation; non-string values are passed through as-is.
        
        Parameters:
            prediction: Model output to evaluate (commonly a string).
            expected: Ground-truth value to compare against.
            sample: Original sample dictionary; passed to metric implementations for context.
            error: Optional error message; when present causes an early error-metrics return.
            error_type: Optional error classification included in error metrics.
        
        Returns:
            A dictionary containing:
              - `valid` (bool): True when metrics were computed, False for error cases.
              - per-metric scores keyed by metric name (or metric output keys when a metric returns a dict).
              - when `valid` is False, includes `error` and `error_type` fields as provided.
        """
        if error:
            return self.get_error_metrics(error, error_type)

        # Normalize if configured
        if isinstance(prediction, str) and isinstance(expected, str):
            pred_normalized = self._normalize_text(prediction)
            exp_normalized = self._normalize_text(expected)
        else:
            pred_normalized = prediction
            exp_normalized = expected

        # Calculate metrics using configured metric objects
        metrics_result = {"valid": True}

        for metric in self.metrics:
            try:
                # Use per_sample() for metric protocol, or compute() for ScalarMetric subclasses
                if hasattr(metric, 'per_sample'):
                    metric_output = metric.per_sample(pred_normalized, exp_normalized, sample)
                elif hasattr(metric, 'compute'):
                    metric_output = metric.compute(pred_normalized, exp_normalized, sample)
                else:
                    continue
                    
                if isinstance(metric_output, dict):
                    metrics_result.update(metric_output)
                else:
                    # Single value metric
                    metric_name = getattr(metric, "name", metric.__class__.__name__.lower())
                    metrics_result[metric_name] = metric_output
            except Exception as e:
                logger.warning(f"Error calculating {metric.__class__.__name__}: {e}")
                continue

        return metrics_result

    def get_error_metrics(
        self, error: str, error_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Builds a standardized metrics dictionary representing a failed prediction.
        
        Parameters:
            error (str): Human-readable error message describing the failure.
            error_type (Optional[str]): Optional error category or code.
        
        Returns:
            Dict[str, Any]: A metrics dictionary with `valid` set to False, `error` and `error_type` populated, and each configured metric name mapped to 0.0.
        """
        error_metrics = {
            "valid": False,
            "error": error,
            "error_type": error_type,
        }

        # Add zero values for all configured metrics
        for metric in self.metrics:
            metric_name = getattr(metric, "name", metric.__class__.__name__.lower())
            error_metrics[metric_name] = 0.0

        return error_metrics

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate per-sample freeform metrics into dataset-level statistics.
        
        Parameters:
            all_metrics (List[Dict]): List of per-sample metric dictionaries. Each dictionary should include a "valid" boolean (defaults to True if missing) and numeric metric entries keyed by metric name (e.g., "exactmatch", "f1score").
        
        Returns:
            Dict[str, Any]: Aggregated metrics including:
                - total_samples: total number of samples processed.
                - valid_samples: number of samples marked valid.
                - error_count: number of invalid samples.
                - error_rate: ratio of invalid samples to total_samples.
                - <metric_name>: average value of each numeric metric computed over valid samples.
        """
        if not all_metrics:
            return {}

        total_samples = len(all_metrics)
        valid_samples = sum(1 for m in all_metrics if m.get("valid", True))

        aggregated = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "error_count": total_samples - valid_samples,
            "error_rate": (total_samples - valid_samples) / total_samples if total_samples > 0 else 0.0,
        }

        # Collect all metric names from the metrics objects
        metric_names = set()
        for metric in self.metrics:
            metric_name = getattr(metric, "name", metric.__class__.__name__.lower())
            metric_names.add(metric_name)

        # Compute averages for each metric
        for metric_name in metric_names:
            values = [m.get(metric_name, 0) for m in all_metrics if m.get("valid", True)]
            if values:
                aggregated[metric_name] = sum(values) / len(values)

        return aggregated
