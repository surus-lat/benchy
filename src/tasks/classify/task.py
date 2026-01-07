"""Classification task implementation built on SimpleTask.

This task relies on SimpleTask for dataset loading, caching, and baseline metric aggregation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from ...common import format_choices
from ..metrics import MultipleChoiceAccuracy
from ..simple_task import SimpleTask

logger = logging.getLogger(__name__)


class ClassifyTask(SimpleTask):
    """SimpleTask-based classification for multiple subtasks.

    The task expects dataset configs to include label_values (optional label_texts)
    so that choices and label mapping can be generated consistently.
    """

    name = "classify"
    metrics = [MultipleChoiceAccuracy()]
    answer_type = "multiple_choice"
    requires_logprobs = False
    prefers_logprobs = True

    def __init__(self, config: Dict[str, Any]):
        # Store subtask name early so we can name the cached dataset file.
        self.subtask_name = config.get("subtask_name", "classify")
        self.default_data_file = f"{self.subtask_name}.jsonl"

        # Capture dataset field mappings before SimpleTask uses the config.
        dataset_config = config.get("dataset", {})
        self.text_field = dataset_config.get("text_field", "text")
        self.label_field = dataset_config.get("label_field", "label")

        self._label_value_type = "numeric"
        self.label_values = self._normalize_label_values(dataset_config.get("label_values"))
        self.label_texts = self._normalize_label_texts(
            dataset_config.get("label_texts") or dataset_config.get("label_map")
        )

        # Build choice lists once so samples can reuse them.
        (
            self.label_to_index,
            self.choice_texts,
            self.choice_labels,
        ) = self._build_choices()

        super().__init__(config)

        if not self.label_values:
            raise ValueError("ClassifyTask requires label_values in the dataset config.")

    def _normalize_label_values(self, label_values: Any) -> Optional[list]:
        """Normalize label values to ints or strings."""
        if not isinstance(label_values, list) or not label_values:
            return None

        normalized = []
        numeric = True
        for value in label_values:
            try:
                normalized.append(int(value))
            except (TypeError, ValueError):
                numeric = False
                break

        if numeric:
            self._label_value_type = "numeric"
            return normalized

        self._label_value_type = "text"
        return [str(value) for value in label_values]

    def _normalize_label_texts(self, label_texts: Any) -> Dict[Any, str]:
        """Normalize label display text mapping."""
        mapping: Dict[Any, str] = {}
        if not label_texts:
            return mapping

        if isinstance(label_texts, list):
            for idx, text in enumerate(label_texts):
                if text is None:
                    continue
                label_value = self.label_values[idx] if self.label_values and idx < len(self.label_values) else idx
                mapping[label_value] = str(text)
            return mapping

        if isinstance(label_texts, dict):
            for key, value in label_texts.items():
                label_value = self._coerce_label_value(key)
                if label_value is None:
                    continue
                mapping[label_value] = str(value)
        return mapping

    def _build_choices(self) -> Tuple[Dict[Any, int], list, list]:
        """Build choice texts/labels and a label-to-index map."""
        label_to_index: Dict[Any, int] = {}
        choice_texts: list = []
        choice_labels: list = []

        if not self.label_values:
            return label_to_index, choice_texts, choice_labels

        for idx, value in enumerate(self.label_values):
            label_to_index[value] = idx
            choice_texts.append(self.label_texts.get(value, str(value)))
            choice_labels.append(str(value))

        return label_to_index, choice_texts, choice_labels

    def _coerce_label_value(self, value: Any) -> Optional[Any]:
        """Coerce raw labels into normalized value types."""
        if value is None:
            return None
        if self._label_value_type == "numeric":
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        return str(value)

    def preprocess_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """Convert a raw dataset record into the eval format."""
        text = raw_sample.get(self.text_field)
        label_raw = raw_sample.get(self.label_field)
        if text is None or label_raw is None:
            return None

        label_value = self._coerce_label_value(label_raw)
        if self.label_values and label_value not in self.label_values:
            return None

        expected_idx = self.label_to_index.get(label_value)
        if expected_idx is None:
            return None

        return {
            "id": f"{self.subtask_name}_{idx}",
            "text": str(text),
            "expected": expected_idx,
            "choices": list(self.choice_texts),
            "choice_labels": list(self.choice_labels),
            "label_to_index": dict(self.label_to_index),
            "label_value": label_value,
        }

    def get_prompt(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Build the system/user prompt for the classification task."""
        prompts = self.config.get("prompts", {})
        system_prompt = prompts.get("system", "You are a helpful assistant.")
        user_template = prompts.get("user", "{text}")

        # Provide formatted choices if the prompt template uses {choices}.
        choices_text = format_choices(sample.get("choices", []), sample.get("choice_labels"))
        user_prompt = user_template.format(text=sample.get("text", ""), choices=choices_text)

        return system_prompt, user_prompt

    def get_prompt_for_logprobs(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Prompt variant tuned for logprobs scoring."""
        system_prompt, user_prompt = self.get_prompt(sample)

        # Ensure choices are visible and the answer format is constrained.
        if "Answer" not in user_prompt and "Respuesta" not in user_prompt:
            choices_text = format_choices(sample.get("choices", []), sample.get("choice_labels"))
            if choices_text and "Options" not in user_prompt and "Opciones" not in user_prompt:
                user_prompt = f"{user_prompt}\n\nOptions:\n{choices_text}"
            user_prompt = f"{user_prompt}\n\nAnswer (label only):"

        return system_prompt, user_prompt

    def get_task_name(self) -> str:
        """Return subtask-specific task name for output separation."""
        return f"classify_{self.subtask_name}"

    def get_error_metrics(self, error: str, error_type: Optional[str] = None) -> Dict[str, Any]:
        """Return a consistent error metrics payload."""
        return {
            "valid": False,
            "error": error,
            "error_type": error_type,
            "accuracy": 0.0,
            "correct": False,
        }
