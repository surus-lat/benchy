"""Structured output format handler.

This handler encapsulates logic for structured extraction tasks including:
- JSON schema validation
- Field-level comparison and metrics
- Partial matching for near-matches
- Hallucination detection
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseHandler

logger = logging.getLogger(__name__)


class StructuredHandler(BaseHandler):
    """Handler for structured extraction tasks.
    
    This handler provides comprehensive support for structured output evaluation including
    JSON schema validation, field-level metrics, and configurable partial matching.
    
    Class Attributes (in addition to BaseHandler):
        requires_schema: Whether schema is required (default: True)
        schema_field: Field name for schema in samples (default: "schema")
        metrics_config: Configuration for metrics calculator (optional)
    
    Example:
        class MyStructuredTask(StructuredHandler):
            dataset = "org/extraction-dataset"
            system_prompt = "You are a data extraction expert."
            metrics_config = {
                "partial_matching": {
                    "string": {"exact_threshold": 0.95},
                    "number": {"relative_tolerance": 0.001}
                }
            }
    """

    # Structured output defaults
    answer_type = "structured"
    requires_schema = True
    schema_field: str = "schema"
    metrics_config: Optional[Dict[str, Any]] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the StructuredHandler and prepare lazy initialization for the metrics calculator.
        
        Parameters:
            config (Optional[Dict[str, Any]]): Optional handler configuration used by the base handler.
        """
        super().__init__(config)

        # Lazy initialization of metrics calculator
        self._metrics_calc = None

    @property
    def metrics_calculator(self):
        """
        Lazily initialize and return the MetricsCalculator used for structured extraction evaluation.
        
        Builds the metrics configuration from the handler's `metrics_config` or `config['metrics']`, reads the `strict` flag from that configuration, constructs a MetricsCalculator with the assembled settings, caches it on the instance, and returns it.
        
        Returns:
            MetricsCalculator: An instance configured for this handler.
        """
        if self._metrics_calc is None:
            from .utils.structured_metrics_calculator import MetricsCalculator

            # Build metrics config
            metrics_cfg = {}
            if self.metrics_config:
                metrics_cfg.update(self.metrics_config)
            elif self.config and "metrics" in self.config:
                metrics_cfg.update(self.config["metrics"])

            # Check if strict mode is enabled
            strict = metrics_cfg.get("strict", False)
            
            self._metrics_calc = MetricsCalculator({"metrics": metrics_cfg}, strict=strict)
        return self._metrics_calc

    def preprocess_sample(
        self, raw_sample: Dict[str, Any], idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a raw dataset sample into the handler's evaluation-ready sample dictionary.
        
        If any of the required fields (text, schema, or expected label) is missing, the sample is skipped and the function returns `None` (a warning is logged).
        
        Parameters:
            raw_sample (Dict[str, Any]): Original dataset sample; expected to contain keys accessible via this handler's `text_field`, `schema_field`, and `label_field`.
            idx (int): Sample index used to construct the returned sample `id`.
        
        Returns:
            Optional[Dict[str, Any]]: A processed sample with keys:
                - `id` (str): "<task_name>_<idx>"
                - `text` (str): Stringified text content
                - `schema` (Any): Extracted schema object
                - `expected` (Any): Expected output/label
            Returns `None` if required fields are missing.
        """
        text = raw_sample.get(self.text_field)
        schema = raw_sample.get(self.schema_field)
        expected = raw_sample.get(self.label_field)

        if text is None or schema is None or expected is None:
            logger.warning(
                f"Skipping sample {idx}: missing text, schema, or expected output"
            )
            return None

        return {
            "id": f"{self.get_task_name()}_{idx}",
            "text": str(text),
            "schema": schema,
            "expected": expected,
        }

    def get_prompt(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """
        Constructs the system and user prompts for a structured-extraction sample.
        
        If the sample contains a schema, the schema is serialized and included in the user prompt.
        If the handler defines a `user_prompt_template` that contains `{schema}`, that template is used
        and formatted with `text` and the serialized `schema`; otherwise a default prompt embedding the
        text and schema is produced.
        
        Parameters:
            sample (Dict[str, Any]): A sample dictionary containing at least `text` and optionally `schema`.
        
        Returns:
            A pair of strings: the system prompt and the user prompt. The user prompt contains the sample text
            and the schema (if present) and instructs the model to return JSON matching the schema.
        """
        schema = sample.get("schema", {})
        schema_str = json.dumps(schema, indent=2) if schema else ""

        # Use custom template if provided
        if hasattr(self, "user_prompt_template") and "{schema}" in self.user_prompt_template:
            user_prompt = self.user_prompt_template.format(
                text=sample.get("text", ""), schema=schema_str
            )
        else:
            # Default structured extraction format
            user_prompt = (
                f"{sample.get('text', '')}\n\n"
                f"Extract information following this schema:\n"
                f"{schema_str}\n\n"
                f"Return valid JSON matching the schema exactly."
            )

        return self.system_prompt, user_prompt

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict[str, Any],
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics for a structured extraction prediction.
        
        Uses the sample's schema (sample.get("schema", {})) when present to validate outputs and produce comprehensive metrics such as schema validity, field-level scores, hallucination rates, and aggregate extraction quality.
        
        Parameters:
            prediction: Model output to evaluate (typically a dict or JSON-like structure).
            expected: Ground-truth output for the sample.
            sample: Full sample dictionary; the function will read the schema from sample.get("schema", {}).
            error: Optional error message associated with the prediction.
            error_type: Optional classification of the error.
        
        Returns:
            dict: Mapping of metric names to values (e.g., schema validity, exact match, field F1 scores, hallucination rate, extraction quality score).
        """
        return self.metrics_calculator.calculate_all(
            prediction=prediction,
            expected=expected,
            schema=sample.get("schema", {}),
            error=error,
            error_type=error_type,
        )

    def get_error_metrics(
        self, error: str, error_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute metrics for a failed prediction using the provided error information.
        
        Parameters:
            error (str): Error message describing the failure.
            error_type (Optional[str]): Optional categorical label for the error.
        
        Returns:
            Dict[str, Any]: Metrics dictionary consistent with successful prediction metrics.
        """
        return self.metrics_calculator.calculate_all(
            prediction=None,
            expected={},
            schema={},
            error=error,
            error_type=error_type,
        )

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate per-sample structured extraction metrics into overall counts, rates, and averaged numeric scores.
        
        Parameters:
            all_metrics (List[Dict]): List of per-sample metric dictionaries produced by metric calculation.
        
        Returns:
            Dict[str, Any]: Aggregated metrics including:
                - total_samples: total number of samples processed.
                - valid_samples: number of samples considered valid.
                - error_count: total_samples minus valid_samples.
                - If there is at least one valid sample, also includes:
                    - schema_validity_rate: fraction of samples with schema_validity > 0.5.
                    - exact_match_rate: fraction of samples with exact_match == True.
                    - error_rate: fraction of samples that are errors.
                    - Averages for numeric metrics (when present for valid samples): 
                      "field_f1_strict", "field_f1_partial", "field_precision",
                      "field_recall", "hallucination_rate", "extraction_quality_score".
        """
        if not all_metrics:
            return {}

        total_samples = len(all_metrics)
        valid_samples = sum(1 for m in all_metrics if m.get("valid", False))

        aggregated = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "error_count": total_samples - valid_samples,
        }

        if valid_samples == 0:
            return aggregated

        # Compute rates for boolean metrics
        schema_valid_count = sum(1 for m in all_metrics if m.get("schema_validity", 0) > 0.5)
        exact_match_count = sum(1 for m in all_metrics if m.get("exact_match", False))

        aggregated["schema_validity_rate"] = schema_valid_count / total_samples
        aggregated["exact_match_rate"] = exact_match_count / total_samples
        aggregated["error_rate"] = (total_samples - valid_samples) / total_samples

        # Compute averages for numeric metrics
        numeric_metrics = [
            "field_f1_strict",
            "field_f1_partial",
            "field_precision",
            "field_recall",
            "hallucination_rate",
            "extraction_quality_score",
        ]

        for metric_name in numeric_metrics:
            values = [m.get(metric_name, 0) for m in all_metrics if m.get("valid", False)]
            if values:
                aggregated[metric_name] = sum(values) / len(values)

        return aggregated
