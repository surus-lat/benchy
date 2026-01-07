"""Multiple choice format handler.

This handler encapsulates common logic for multiple-choice tasks including:
- Label mapping and choice generation
- Logprobs-based scoring when available
- Standard accuracy metrics
- Fallback to text parsing when logprobs unavailable
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from ...common import format_choices, parse_choice_index
from ..metrics import MultipleChoiceAccuracy
from .base import BaseHandler

logger = logging.getLogger(__name__)


class MultipleChoiceHandler(BaseHandler):
    """Handler for multiple-choice tasks.
    
    This handler provides comprehensive support for multiple-choice evaluation including
    label mapping, choice formatting, logprobs-based scoring, and accuracy metrics.
    
    Class Attributes (in addition to BaseHandler):
        labels: Dict mapping label values to display text (e.g., {0: "No", 1: "Yes"})
        label_field: Field name for the label in raw data (default: "label")
        choice_labels_field: Optional field for custom choice labels
        prefers_logprobs: Whether to prefer logprobs scoring (default: True)
    
    Example:
        class MyMCQTask(MultipleChoiceHandler):
            dataset = "org/my-dataset"
            labels = {0: "No", 1: "Yes", 2: "Maybe"}
            system_prompt = "You are a classifier."
    """

    # Multiple choice defaults
    answer_type = "multiple_choice"
    prefers_logprobs = True
    choice_labels_field: Optional[str] = None

    # Required attributes (must be set in subclass)
    labels: Dict[Any, str] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the multiple choice handler."""
        super().__init__(config)

        # Validate labels
        if not self.labels:
            raise ValueError(
                f"{self.__class__.__name__} requires 'labels' attribute "
                "(e.g., labels = {{0: 'No', 1: 'Yes'}})"
            )

        # Build choice mapping
        self.label_to_index = {}
        self.choice_texts = []
        self.choice_labels = []

        for idx, (value, text) in enumerate(sorted(self.labels.items())):
            self.label_to_index[value] = idx
            self.choice_texts.append(text)
            self.choice_labels.append(str(value))

        # Set default metrics
        if not self.metrics:
            self.metrics = [MultipleChoiceAccuracy()]

    def preprocess_sample(
        self, raw_sample: Dict[str, Any], idx: int
    ) -> Optional[Dict[str, Any]]:
        """Transform a raw dataset sample to eval format.
        
        Extracts text and label, maps label to choice index, and attaches
        choice information.
        
        Args:
            raw_sample: Raw sample from dataset
            idx: Sample index
            
        Returns:
            Processed sample with choices, expected index, etc.
        """
        text = raw_sample.get(self.text_field)
        label_raw = raw_sample.get(self.label_field)

        if text is None or label_raw is None:
            logger.warning(f"Skipping sample {idx}: missing text or label")
            return None

        # Coerce label to correct type
        label_value = self._coerce_label(label_raw)
        if label_value not in self.label_to_index:
            logger.warning(
                f"Skipping sample {idx}: label {label_raw} not in labels mapping"
            )
            return None

        expected_idx = self.label_to_index[label_value]

        return {
            "id": f"{self.get_task_name()}_{idx}",
            "text": str(text),
            "expected": expected_idx,
            "choices": list(self.choice_texts),
            "choice_labels": list(self.choice_labels),
            "label_to_index": dict(self.label_to_index),
            "label_value": label_value,
        }

    def _coerce_label(self, value: Any) -> Any:
        """Coerce label to the correct type based on labels dict keys.
        
        Args:
            value: Raw label value
            
        Returns:
            Coerced label value matching labels dict key type
        """
        if value is None:
            return None

        # Determine type from first label key
        first_key = next(iter(self.labels.keys()))
        if isinstance(first_key, int):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        return str(value)

    def get_prompt(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Build prompts for multiple choice tasks.
        
        Formats choices and constructs a prompt that works well for both
        completion-based and logprobs-based scoring.
        
        Args:
            sample: Sample with text, choices, choice_labels
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        choices_text = format_choices(
            sample.get("choices", []), sample.get("choice_labels")
        )

        # Use custom template if provided, otherwise default format
        if hasattr(self, "user_prompt_template") and "{choices}" in self.user_prompt_template:
            user_prompt = self.user_prompt_template.format(
                text=sample.get("text", ""), choices=choices_text
            )
        else:
            # Default multiple choice format
            user_prompt = (
                f"{sample.get('text', '')}\n\n"
                f"Options:\n{choices_text}\n\n"
                f"Answer (label only):"
            )

        return self.system_prompt, user_prompt

    def get_prompt_for_logprobs(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Build prompts optimized for logprobs scoring.
        
        Ensures choices are clearly visible and answer format is constrained.
        This method is called by interfaces that support logprobs.
        
        Args:
            sample: Sample dict
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Get base prompt
        system_prompt, user_prompt = self.get_prompt(sample)

        # Ensure answer format is clear
        if "Answer" not in user_prompt and "answer" not in user_prompt.lower():
            user_prompt = f"{user_prompt}\n\nAnswer (label only):"

        return system_prompt, user_prompt

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict[str, Any],
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for multiple choice prediction.
        
        Handles both logprobs-based predictions (already an index) and
        text-based predictions (need parsing).
        
        Args:
            prediction: Model output (choice index or text)
            expected: Expected choice index
            sample: Full sample dict
            error: Error message if any
            error_type: Type of error
            
        Returns:
            Metrics dict with accuracy and correctness
        """
        if error:
            return self.get_error_metrics(error, error_type)

        # Parse prediction if it's text
        if isinstance(prediction, str):
            prediction_idx = parse_choice_index(
                prediction, sample.get("choice_labels", [])
            )
        else:
            prediction_idx = prediction

        # Calculate correctness
        correct = prediction_idx == expected if prediction_idx is not None else False

        # Use metrics from class or default
        metrics_result = {"valid": True, "correct": correct, "accuracy": 1.0 if correct else 0.0}

        # Run additional metrics if defined
        for metric in self.metrics:
            metric_output = metric.compute(prediction_idx, expected, sample)
            if isinstance(metric_output, dict):
                metrics_result.update(metric_output)
            else:
                metrics_result[metric.name] = metric_output

        return metrics_result

    def get_error_metrics(
        self, error: str, error_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return error metrics for failed predictions.
        
        Args:
            error: Error message
            error_type: Type of error
            
        Returns:
            Metrics dict matching structure of successful predictions
        """
        return {
            "valid": False,
            "error": error,
            "error_type": error_type,
            "correct": False,
            "accuracy": 0.0,
        }

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate multiple choice metrics across all samples.
        
        Computes overall accuracy and error rates.
        
        Args:
            all_metrics: List of per-sample metrics
            
        Returns:
            Aggregated metrics dict
        """
        if not all_metrics:
            return {}

        total_samples = len(all_metrics)
        valid_samples = sum(1 for m in all_metrics if m.get("valid", True))
        correct_samples = sum(1 for m in all_metrics if m.get("correct", False))

        aggregated = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "error_count": total_samples - valid_samples,
            "accuracy": correct_samples / valid_samples if valid_samples > 0 else 0.0,
            "error_rate": (total_samples - valid_samples) / total_samples if total_samples > 0 else 0.0,
        }

        return aggregated

