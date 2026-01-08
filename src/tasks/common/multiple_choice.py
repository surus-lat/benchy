"""Multiple choice format handler.

This handler encapsulates common logic for multiple-choice tasks including:
- Label mapping and choice generation
- Logprobs-based scoring when available
- Standard accuracy metrics
- Fallback to text parsing when logprobs unavailable

IMPORTANT: Understanding the `labels` Attribute
================================================

The `labels` attribute is OPTIONAL and controls how choices are handled.
There are two modes of operation:

MODE 1: Task-Level Labels (Simple Tasks)
-----------------------------------------
Use this when ALL samples share the same choices.

Example:
    class YesNoTask(MultipleChoiceHandler):
        labels = {0: "No", 1: "Yes"}  # All samples use these choices
        dataset_name = "my/dataset"
        
In this mode:
- The handler extracts `text` and `label` from each sample
- It automatically adds `choices`, `choice_labels`, and `expected` index
- The `labels` dict maps dataset label values to choice text
- Works with datasets that have a simple label column (0/1, "yes"/"no", etc.)

MODE 2: Per-Sample Choices (Complex Tasks)
-------------------------------------------
Use this when choices vary per sample (e.g., different questions have different options).

Example:
    class ExamTask(MultipleChoiceHandler):
        # NO labels attribute!
        dataset_name = "my/exam-dataset"
        
        def _download_and_cache(self, output_path):
            raw = download_huggingface_dataset(self.dataset_name)
            processed = []
            for sample in raw:
                processed.append({
                    "id": sample["id"],
                    "text": sample["question"],
                    "choices": ["Option A", "Option B", "Option C"],  # Per-sample!
                    "choice_labels": ["A", "B", "C"],
                    "expected": sample["correct_idx"],
                })
            save_to_jsonl(processed, output_path)

In this mode:
- Each sample provides its own `choices`, `choice_labels`, and `expected`
- The `labels` attribute is NOT needed
- You have full control over choice formatting per sample
- Useful for exams where questions have different numbers of choices

When to Use Which Mode?
------------------------
Task-Level Labels (MODE 1):
  ✅ All samples have the same set of choices
  ✅ Dataset has a simple label field (0/1, "positive"/"negative", etc.)
  ✅ You want minimal code
  
Per-Sample Choices (MODE 2):
  ✅ Different samples have different choices
  ✅ Different numbers of choices per sample
  ✅ Choices need custom formatting per sample
  ✅ You're downloading from HuggingFace with complex structure

Error Messages
--------------
If you see: "requires 'labels' attribute":
  → You're in MODE 1 but forgot to define labels
  → Add: labels = {0: "Choice1", 1: "Choice2"}

If you see: "missing 'choices'":
  → You're in MODE 2 but samples don't have choices
  → Add choices/choice_labels/expected in _download_and_cache()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from .utils.choice_utils import format_choices, parse_choice_prediction
from .metrics import MultipleChoiceAccuracy
from .base import BaseHandler

logger = logging.getLogger(__name__)


class MultipleChoiceHandler(BaseHandler):
    """Handler for multiple-choice tasks.
    
    This handler provides comprehensive support for multiple-choice evaluation including
    label mapping, choice formatting, logprobs-based scoring, and accuracy metrics.
    
    Class Attributes (in addition to BaseHandler):
        labels: OPTIONAL dict mapping label values to choice text
                Example: {0: "No", 1: "Yes"}
                - Set this for simple tasks where all samples share the same choices
                - Omit this for complex tasks where samples provide their own choices
                See module docstring for detailed explanation of the two modes.
        
        label_field: Field name for the label in raw data (default: "label")
                     Only used in MODE 1 (task-level labels)
        
        text_field: Field name for the question text (default: "text")
                    Only used in MODE 1 (task-level labels)
        
        choice_labels_field: Optional field for custom choice labels
        
        prefers_logprobs: Whether to prefer logprobs scoring (default: True)
                          When True, uses first-token log probabilities for prediction
                          When False or unavailable, falls back to text parsing
    
    Examples:
        # MODE 1: Task-level labels (simple)
        class SentimentTask(MultipleChoiceHandler):
            dataset_name = "sentiment/dataset"
            labels = {0: "Negative", 1: "Positive"}  # Required!
            # Handler extracts text/label and adds choices automatically
        
        # MODE 2: Per-sample choices (complex)
        class ExamTask(CachedDatasetMixin, MultipleChoiceHandler):
            dataset_name = "exam/dataset"
            # No labels! Samples provide their own choices
            
            def _download_and_cache(self, output_path):
                raw = download_huggingface_dataset(self.dataset_name)
                processed = []
                for sample in raw:
                    processed.append({
                        "text": sample["question"],
                        "choices": sample["options"],  # Per-sample!
                        "expected": sample["answer_idx"],
                    })
                save_to_jsonl(processed, output_path)
    """

    # Multiple choice defaults
    answer_type = "multiple_choice"
    prefers_logprobs = True
    choice_labels_field: Optional[str] = None
    system_prompt = "You will only answer with the correct choice. No other text or explanation is allowed."
    
    # Parsing strictness (set to False for more permissive matching)
    strict_parsing: bool = True

    # Optional task-level labels
    # IMPORTANT: See class and module docstrings for when to use this!
    # - Set for simple tasks: labels = {0: "No", 1: "Yes"}
    # - Omit for complex tasks where samples provide their own choices
    labels: Dict[Any, str] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the handler and prepare label-to-choice mappings.
        
        Builds internal mappings from `self.labels` (when present) into `self.label_to_index`,
        `self.choice_texts`, and `self.choice_labels` so the handler can operate in task-level
        labels mode. If no metrics are configured, assigns a default `[MultipleChoiceAccuracy()]`.
        
        Parameters:
        	config (Optional[Dict[str, Any]]): Optional configuration passed to the base handler.
        """
        super().__init__(config)

        # Build choice mapping from labels if provided
        # If not provided, samples must provide their own choice_labels
        self.label_to_index = {}
        self.choice_texts = []
        self.choice_labels = []

        if self.labels:
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
        """
        Preprocess a raw dataset sample into the evaluation format for multiple-choice tasks.
        
        Supports two modes:
        - Task-level labels mode: when the handler has a `labels` mapping, extracts the sample text and label (using `text_field` and `label_field`), converts the label to the corresponding choice index, and attaches the task's `choices`, `choice_labels`, and `label_to_index`.
        - Per-sample choices passthrough: if the sample already contains `choices` and `expected`, returns a copy (adds an `id` if missing).
        
        Parameters:
            raw_sample (Dict[str, Any]): The original dataset sample.
            idx (int): Index of the sample (used to build a default `id` when needed).
        
        Returns:
            Optional[Dict[str, Any]]: A processed sample containing at minimum:
                - "id": sample identifier
                - "text": prompt text
                - "choices": list of choice texts
                - "choice_labels": list of choice labels
                - "expected": index of the correct choice
                - "label_to_index": mapping from label values to indices (when produced)
                - "label_value": original label value (when produced)
            Returns None if the sample is missing required fields or its label is not in the handler's mapping.
        
        Raises:
            ValueError: If operating in task-level labels mode but the handler has no `labels` configured.
        """
        # MODE 2: If sample already has choices and expected, it's preprocessed
        if "choices" in raw_sample and "expected" in raw_sample:
            sample = dict(raw_sample)
            if "id" not in sample:
                sample["id"] = f"{self.get_task_name()}_{idx}"
            return sample
        
        # MODE 1: Extract from text_field and label_field, use task-level labels
        text = raw_sample.get(self.text_field)
        label_raw = raw_sample.get(self.label_field)

        if text is None or label_raw is None:
            logger.warning(f"Skipping sample {idx}: missing text or label")
            return None

        # Validate we have labels to map to
        if not self.label_to_index:
            raise ValueError(
                f"\n{self.__class__.__name__} is in MODE 1 (extracting from {self.text_field}/{self.label_field}) "
                f"but 'labels' attribute is not set.\n\n"
                f"Fix this by either:\n"
                f"  1. Add task-level labels: labels = {{0: 'Choice1', 1: 'Choice2'}}\n"
                f"  2. Preprocess samples in _download_and_cache() to include 'choices' and 'expected'\n\n"
                f"See MultipleChoiceHandler docstring for detailed explanation of the two modes."
            )

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
        """
        Convert a raw label value to the same type used by the handler's label keys.
        
        Parameters:
            value (Any): Raw label value to coerce.
        
        Returns:
            Any: The coerced label value with the same type as the keys in `self.labels`, or `None` if `value` is `None` or cannot be converted.
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
        """
        Constructs system and user prompts for a multiple-choice sample.
        
        Formats the sample's choices and inserts them into a user prompt. If the handler defines a `user_prompt_template` containing "{choices}", that template is used with `text` and `choices` substitutions; otherwise a default prompt is created that includes the sample text, an "Options:" list, and the directive "Answer (label only):".
        
        Parameters:
            sample (Dict[str, Any]): Sample containing at least `text`, and optionally `choices` and `choice_labels`.
        
        Returns:
            Tuple[str, str]: `(system_prompt, user_prompt)` strings to be used with the model.
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
        """
        Produce system and user prompts tailored for logprobs-based scoring.
        
        If the user prompt does not already include an answer directive, appends "Answer (label only):" to constrain model output for logprobs evaluation.
        
        Parameters:
            sample (Dict[str, Any]): Preprocessed sample dictionary used to build the prompt.
        
        Returns:
            Tuple[str, str]: A tuple of (system_prompt, user_prompt).
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
        """
        Evaluate a single multiple-choice prediction and return per-sample metrics.
        
        Parses the model's prediction into a choice index, compares it to the expected index, and computes correctness, accuracy, and any additional configured per-sample metrics. If `error` is provided, returns a standardized error metrics dict.
        
        Parameters:
            prediction: Model output to evaluate; may be a choice index or text that can be parsed into a choice.
            expected: The expected choice index for the sample.
            sample: The sample dictionary (should contain `choices`, and optionally `choice_labels` and `label_to_index`) used for parsing and metric computation.
            error: Optional error message; when present, triggers error metrics.
            error_type: Optional classification of the error.
        
        Returns:
            A dict containing at minimum:
              - `valid` (bool): whether the prediction was processed as valid,
              - `correct` (bool): whether the parsed prediction matches `expected`,
              - `accuracy` (float): 1.0 for correct, 0.0 for incorrect,
            plus any additional metric outputs from configured metrics.
        """
        if error:
            return self.get_error_metrics(error, error_type)

        # Use choice_utils for robust parsing (with strict mode control)
        prediction_idx = parse_choice_prediction(
            prediction,
            sample.get("choices", []),
            labels=sample.get("choice_labels"),
            label_to_index=sample.get("label_to_index"),
            strict=self.strict_parsing,
        )

        # Calculate correctness
        correct = prediction_idx == expected if prediction_idx is not None else False

        # Use metrics from class or default
        metrics_result = {"valid": True, "correct": correct, "accuracy": 1.0 if correct else 0.0}

        # Run additional metrics if defined
        for metric in self.metrics:
            # Use per_sample() for metric protocol, or compute() for ScalarMetric subclasses
            if hasattr(metric, 'per_sample'):
                metric_output = metric.per_sample(prediction_idx, expected, sample)
            elif hasattr(metric, 'compute'):
                metric_output = metric.compute(prediction_idx, expected, sample)
            else:
                continue
                
            if isinstance(metric_output, dict):
                metrics_result.update(metric_output)
            else:
                metrics_result[metric.name] = metric_output

        return metrics_result

    def get_error_metrics(
        self, error: str, error_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Produce a standardized metrics dict for a prediction that failed due to an error.
        
        Parameters:
            error (str): Error message describing the failure.
            error_type (Optional[str]): Optional classification of the error.
        
        Returns:
            dict: Metrics with keys:
                - valid: False
                - error: the provided error message
                - error_type: the provided error type or None
                - correct: False
                - accuracy: 0.0
        """
        return {
            "valid": False,
            "error": error,
            "error_type": error_type,
            "correct": False,
            "accuracy": 0.0,
        }

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate per-sample multiple-choice metrics into overall statistics.
        
        Parameters:
            all_metrics (List[Dict]): List of per-sample metric dictionaries; each dictionary may include the boolean keys `valid` and `correct`.
        
        Returns:
            Dict[str, Any]: Aggregated metrics with keys:
                - total_samples: total number of samples in `all_metrics`
                - valid_samples: count of samples with `valid` == True (defaults to True if missing)
                - error_count: number of samples considered invalid
                - accuracy: fraction of `correct` samples among valid samples (0.0 if no valid samples)
                - error_rate: fraction of invalid samples among all samples (0.0 if `all_metrics` is empty)
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
