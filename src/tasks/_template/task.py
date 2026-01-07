"""Template task implementation using SimpleTask.

This template includes three subtask formats to make it easy to delete
what you do not need: multiple choice, structured extraction, and freeform.
"""

from typing import Any, Dict, Optional, Tuple

from ...common import format_choices
from ..metrics import ExactMatch, F1Score, MultipleChoiceAccuracy
from ..simple_task import SimpleTask


class TemplateMultipleChoiceTask(SimpleTask):
    """Multiple-choice template task."""

    name = "template_multiple_choice"
    metrics = [MultipleChoiceAccuracy()]
    answer_type = "multiple_choice"
    prefers_logprobs = True

    def preprocess_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """Transform a raw sample into the eval format.

        Expected fields:
        - text: input string
        - choices: list of label options
        - expected: correct choice index
        """
        text = raw_sample.get("text", raw_sample.get("input", ""))
        choices = raw_sample.get("choices")
        choice_labels = raw_sample.get("choice_labels")
        expected = raw_sample.get("expected")
        if text is None or choices is None or expected is None:
            return None
        return {
            "id": f"mcq_{idx}",
            "text": text,
            "choices": choices,
            "choice_labels": choice_labels,
            "expected": expected,
        }

    def get_prompt(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Build a prompt including the choices list."""
        prompts = self.config.get("prompts", {})
        system_prompt = prompts.get("system", "")
        user_template = prompts.get("user", "{text}\n\nOptions:\n{choices}\n\nAnswer:")
        choices_text = format_choices(sample.get("choices", []), sample.get("choice_labels"))
        user_prompt = user_template.format(text=sample.get("text", ""), choices=choices_text)
        return system_prompt, user_prompt


class TemplateStructuredTask(SimpleTask):
    """Structured-output template task.

    Replace ExactMatch with structured metrics in real tasks.
    """

    name = "template_structured"
    metrics = [ExactMatch()]
    requires_schema = True

    def preprocess_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """Transform a raw sample into the eval format."""
        text = raw_sample.get("text")
        schema = raw_sample.get("schema")
        expected = raw_sample.get("expected")
        if text is None or schema is None or expected is None:
            return None
        return {
            "id": f"struct_{idx}",
            "text": text,
            "schema": schema,
            "expected": expected,
        }


class TemplateFreeformTask(SimpleTask):
    """Freeform template task."""

    name = "template_freeform"
    metrics = [F1Score()]

    def preprocess_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """Transform a raw sample into the eval format."""
        text = raw_sample.get("text", raw_sample.get("input", ""))
        expected = raw_sample.get("expected", raw_sample.get("label", raw_sample.get("output")))
        if text is None or expected is None:
            return None
        return {
            "id": f"freeform_{idx}",
            "text": text,
            "expected": expected,
        }
