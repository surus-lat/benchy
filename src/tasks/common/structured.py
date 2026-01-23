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
        """Initialize the structured handler."""
        super().__init__(config)

        # Lazy initialization of metrics calculator
        self._metrics_calc = None

    @property
    def metrics_calculator(self):
        """Lazy initialization of structured extraction metrics calculator.
        
        The real MetricsCalculator from the original structured task,
        preserving EQS calculation and comprehensive error classification.
        
        Returns:
            MetricsCalculator instance
        """
        if self._metrics_calc is None:
            from .utils.structured_metrics_calculator import MetricsCalculator

            # Build metrics config (defaults < config overrides).
            import copy
            metrics_cfg: Dict[str, Any] = copy.deepcopy(self.metrics_config) if self.metrics_config else {}
            if self.config and "metrics" in self.config and isinstance(self.config["metrics"], dict):
                self._deep_merge(metrics_cfg, self.config["metrics"])

            # Check if strict mode is enabled
            strict = metrics_cfg.get("strict", False)
            
            self._metrics_calc = MetricsCalculator({"metrics": metrics_cfg}, strict=strict)
        return self._metrics_calc

    def _deep_merge(self, base: Dict, override: Dict) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def preprocess_sample(
        self, raw_sample: Dict[str, Any], idx: int
    ) -> Optional[Dict[str, Any]]:
        """Transform a raw dataset sample to eval format.
        
        Extracts text, schema, and expected output.
        
        Args:
            raw_sample: Raw sample from dataset
            idx: Sample index
            
        Returns:
            Processed sample with schema and expected fields
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
        """Build prompts for structured extraction tasks.
        
        Includes schema information in the prompt if available.
        
        Args:
            sample: Sample with text and schema
            
        Returns:
            Tuple of (system_prompt, user_prompt)
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
        """Calculate metrics for structured extraction prediction.
        
        Uses the structured extraction MetricsCalculator for comprehensive evaluation
        including schema validation, field-level F1, hallucination detection, etc.
        
        Args:
            prediction: Model output (should be dict/JSON)
            expected: Expected output dict
            sample: Full sample dict with schema
            error: Error message if any
            error_type: Type of error
            
        Returns:
            Comprehensive metrics dict
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
        """Return error metrics for failed predictions.
        
        Args:
            error: Error message
            error_type: Type of error
            
        Returns:
            Metrics dict matching structure of successful predictions
        """
        return self.metrics_calculator.calculate_all(
            prediction=None,
            expected={},
            schema={},
            error=error,
            error_type=error_type,
        )

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate structured extraction metrics across all samples."""
        aggregated = self.metrics_calculator.aggregate_metrics(all_metrics)
        if "field_precision_partial" in aggregated and "field_precision" not in aggregated:
            aggregated["field_precision"] = aggregated.get("field_precision_partial", 0.0)
        if "field_recall_partial" in aggregated and "field_recall" not in aggregated:
            aggregated["field_recall"] = aggregated.get("field_recall_partial", 0.0)
        return aggregated
