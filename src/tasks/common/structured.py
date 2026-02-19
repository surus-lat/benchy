"""Structured output format handler.

This handler encapsulates logic for structured extraction tasks including:
- JSON schema validation
- Field-level comparison and metrics
- Partial matching for near-matches
- Hallucination detection

## CLI Usage

Create structured extraction tasks directly from the CLI:

```bash
# With schema in dataset
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type structured \
  --dataset-name my-org/invoice-extraction \
  --dataset-schema-field schema \
  --limit 10

# With external schema file
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type structured \
  --dataset-name my-org/invoice-extraction \
  --dataset-schema-path schemas/invoice_schema.json \
  --system-prompt "Extract invoice information as JSON." \
  --limit 10

# With inline schema
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type structured \
  --dataset-name ./data/invoices.jsonl \
  --dataset-source local \
  --dataset-schema-json '{"type": "object", "properties": {"name": {"type": "string"}, "amount": {"type": "number"}}, "required": ["name", "amount"]}' \
  --limit 10

# Multimodal structured extraction (e.g., from images)
benchy eval --model-name gpt-4o-mini --provider openai \
  --task-type structured \
  --dataset-name ./data/receipts/ \
  --multimodal-input \
  --dataset-schema-path schemas/receipt_schema.json \
  --limit 10
```

## Dataset Format

**With schema in dataset**:
```jsonl
{"id": "1", "text": "Invoice from...", "schema": {...}, "expected": {...}}
{"id": "2", "text": "Invoice from...", "schema": {...}, "expected": {...}}
```

**With external schema** (use --dataset-schema-path):
```jsonl
{"id": "1", "text": "Invoice from...", "expected": {"invoice_number": "INV-001", "date": "2024-01-15", "total": 1500.00}}
{"id": "2", "text": "Invoice from...", "expected": {"invoice_number": "INV-002", "date": "2024-01-16", "total": 2300.50}}
```

**Schema file format** (JSON):
```json
{
  "type": "object",
  "properties": {
    "invoice_number": {"type": "string"},
    "date": {"type": "string", "format": "date"},
    "total": {"type": "number"},
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "description": {"type": "string"},
          "quantity": {"type": "integer"},
          "price": {"type": "number"}
        }
      }
    }
  },
  "required": ["invoice_number", "date", "total"]
}
```

## Metrics

- **Extraction Quality Score (EQS)**: Primary metric (0-1), weighted by field importance
- **Field-level accuracy**: Per-field exact match rates
- **Schema compliance**: Validation against JSON schema
- **Partial matching**: Credit for near-matches (configurable)
- **Hallucination detection**: Penalties for extra/incorrect fields

## Schema Validation

The handler supports full JSON Schema Draft 7 including:
- Type validation (string, number, integer, boolean, array, object)
- Format validation (date, email, uri, etc.)
- Required fields
- Nested objects and arrays
- Pattern matching (regex)
- Enum constraints
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
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
    field_diagnostics_enabled: bool = True
    field_diagnostics_max_examples_per_field: int = 20
    field_diagnostics_max_fields_in_report: int = 200
    field_diagnostics_max_value_chars: int = 240

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the structured handler."""
        # Apply config overrides BEFORE calling super().__init__
        if config:
            self._apply_config_overrides(config)
        
        super().__init__(config)

        # Lazy initialization of metrics calculator
        self._metrics_calc = None
    
    def _apply_config_overrides(self, config: Dict[str, Any]):
        """Apply configuration overrides to handler attributes.
        
        This allows CLI/config-driven tasks to work with the same handler
        as Python-defined tasks.
        """
        dataset_config = config.get("dataset", {})
        
        # Field mappings
        if "input_field" in dataset_config:
            self.text_field = dataset_config["input_field"]
        if "output_field" in dataset_config:
            self.label_field = dataset_config["output_field"]
        if "schema_field" in dataset_config:
            self.schema_field = dataset_config["schema_field"]
        
        # Schema loading (from path or inline JSON)
        if "schema_path" in dataset_config:
            from pathlib import Path
            schema_path = Path(dataset_config["schema_path"])
            if schema_path.exists():
                with open(schema_path) as f:
                    self._global_schema = json.load(f)
            else:
                self._global_schema = None
                message = (
                    f"Schema file not found: {schema_path} "
                    f"(dataset_config['schema_path']={dataset_config['schema_path']!r})"
                )
                logger.error(message)
                raise FileNotFoundError(message)
        elif "schema_json" in dataset_config:
            self._global_schema = json.loads(dataset_config["schema_json"])
        else:
            self._global_schema = None
        
        # Prompts
        if "system_prompt" in config:
            self.system_prompt = config["system_prompt"]
        if "user_prompt_template" in config:
            self.user_prompt_template = config["user_prompt_template"]
        
        # Multimodal support
        if dataset_config.get("multimodal_input"):
            self.requires_multimodal = True

        diagnostics_cfg = config.get("metrics", {}).get("field_diagnostics", {}) if config else {}
        if isinstance(diagnostics_cfg, dict):
            if "enabled" in diagnostics_cfg:
                self.field_diagnostics_enabled = bool(diagnostics_cfg["enabled"])
            if "max_examples_per_field" in diagnostics_cfg:
                self.field_diagnostics_max_examples_per_field = max(
                    1, int(diagnostics_cfg["max_examples_per_field"])
                )
            if "max_fields_in_report" in diagnostics_cfg:
                self.field_diagnostics_max_fields_in_report = max(
                    1, int(diagnostics_cfg["max_fields_in_report"])
                )
            if "max_value_chars" in diagnostics_cfg:
                self.field_diagnostics_max_value_chars = max(
                    16, int(diagnostics_cfg["max_value_chars"])
                )

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
        Uses global schema if provided via config, otherwise extracts from sample.
        
        Args:
            raw_sample: Raw sample from dataset
            idx: Sample index
            
        Returns:
            Processed sample with schema and expected fields
        """
        # If sample is already preprocessed, return as-is
        if raw_sample.get("_preprocessed") is True and "expected" in raw_sample:
            return raw_sample
        
        text = raw_sample.get(self.text_field)
        expected = raw_sample.get(self.label_field)
        
        # Get schema from global config or sample
        if hasattr(self, '_global_schema') and self._global_schema is not None:
            schema = self._global_schema
        else:
            schema = raw_sample.get(self.schema_field)

        if text is None or expected is None:
            logger.warning(
                f"Skipping sample {idx}: missing text or expected output"
            )
            return None
        
        if schema is None:
            logger.warning(
                f"Skipping sample {idx}: missing schema (not in sample and not in config)"
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

    def build_additional_artifacts(
        self,
        *,
        results: Dict[str, Any],
        output_dir: Path,
        safe_model_name: str,
        timestamp: str,
        task_name: str,
    ) -> List[Path]:
        """Write field-level diagnostics for single-schema structured tasks."""
        if not self.field_diagnostics_enabled:
            return []

        from .utils.field_diagnostics_report import (
            build_field_diagnostics_report,
            write_field_diagnostics_artifacts,
        )

        per_sample_metrics = results.get("per_sample_metrics", [])
        report = build_field_diagnostics_report(
            per_sample_metrics=per_sample_metrics,
            max_examples_per_field=self.field_diagnostics_max_examples_per_field,
            max_fields_in_report=self.field_diagnostics_max_fields_in_report,
            max_value_chars=self.field_diagnostics_max_value_chars,
            require_single_schema=True,
        )

        if report.get("status") != "ok":
            logger.info(
                "Skipping field diagnostics for %s: %s",
                task_name,
                report.get("reason", "unknown"),
            )
            return []

        return write_field_diagnostics_artifacts(
            report=report,
            output_dir=output_dir,
            safe_model_name=safe_model_name,
            timestamp=timestamp,
        )
