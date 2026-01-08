"""Multimodal structured output format handler.

This handler combines multimodal input handling (images, files) with structured
JSON output evaluation. Used for tasks like image extraction, document analysis, etc.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .base import BaseHandler

logger = logging.getLogger(__name__)


class MultimodalStructuredHandler(BaseHandler):
    """Handler for multimodal input to structured output tasks.
    
    This handler provides support for tasks that take images/files as input and
    produce structured JSON output following a schema. It combines file loading,
    schema validation, and structured metrics.
    
    Class Attributes (in addition to BaseHandler):
        input_type: Type of input ("image", "audio", "video", etc.)
        requires_multimodal: Multimodal inputs required (default: True)
        requires_files: File inputs required (default: True)
        requires_schema: Schema required (default: True)
        answer_type: Set to "structured"
        source_dir: Source directory path for files (required)
        schema_field: Field name for schema (default: "schema")
        image_field: Field name for image path (default: "image_path")
        metrics_config: Configuration for metrics calculator (optional)
    
    Example:
        class ImageExtraction(MultimodalStructuredHandler):
            source_dir = "./test_images"
            system_prompt = "Extract data from the image."
            metrics_config = {
                "partial_matching": {
                    "string": {"exact_threshold": 0.85}
                }
            }
    """

    # Multimodal + structured defaults
    input_type: str = "image"
    requires_multimodal: bool = True
    requires_files: bool = True
    requires_schema: bool = True
    answer_type: str = "structured"
    schema_field: str = "schema"
    image_field: str = "image_path"
    source_dir: Optional[str] = None
    metrics_config: Optional[Dict[str, Any]] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the multimodal structured handler."""
        super().__init__(config)

        # Override source_dir from config if provided
        if config and "source_dir" in config:
            self.source_dir = config["source_dir"]

        if not self.source_dir:
            raise ValueError(
                f"{self.__class__.__name__} requires 'source_dir' attribute "
                "pointing to directory with images/files"
            )

        self.source_path = Path(self.source_dir)
        self.schema: Optional[Dict] = None
        self.dataset_metrics_config: Optional[Dict] = None

        # Lazy initialization of metrics calculator
        self._metrics_calc = None

    @property
    def metrics_calculator(self):
        """Lazy initialization of document extraction metrics calculator.
        
        Returns:
            DocumentExtractionMetrics or MetricsCalculator instance
        """
        if self._metrics_calc is None:
            # Use the common MetricsCalculator from utils
            try:
                from .utils.structured_metrics_calculator import MetricsCalculator
                merged_config = self._get_merged_config()
                strict = merged_config.get("strict", False)
                self._metrics_calc = MetricsCalculator({"metrics": merged_config}, strict=strict)
            except ImportError as e:
                logger.error(f"Could not import MetricsCalculator: {e}")
                raise

        return self._metrics_calc

    def _get_merged_config(self) -> Dict:
        """Get merged configuration with dataset-specific metrics config.
        
        Returns:
            Merged configuration dictionary
        """
        import copy

        merged = copy.deepcopy(self.config) if self.config else {}

        # Add class-level metrics config
        if self.metrics_config:
            if "metrics" not in merged:
                merged["metrics"] = {}
            self._deep_merge(merged["metrics"], self.metrics_config)

        # Merge dataset metrics config if available
        if self.dataset_metrics_config:
            if "metrics" not in merged:
                merged["metrics"] = {}
            self._deep_merge(merged["metrics"], self.dataset_metrics_config)

        return merged

    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """Recursively merge override dict into base dict.
        
        Args:
            base: Base dictionary (modified in place)
            override: Override dictionary
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def load(self) -> None:
        """Load dataset, copying from source if needed."""
        # Check if data exists, otherwise copy from source
        if not (self.data_dir / "schema.json").exists():
            if not self.source_path.exists():
                raise FileNotFoundError(
                    f"Source directory not found: {self.source_path}\n"
                    f"Please provide a valid source_dir"
                )
            logger.info(f"Copying data from {self.source_path} to {self.data_dir}")
            self._copy_source_data()

        # Load schema
        schema_file = self.data_dir / "schema.json"
        if schema_file.exists():
            with open(schema_file, "r") as f:
                self.schema = json.load(f)

        # Load dataset metrics config if available
        metrics_config_file = self.data_dir / "metrics_config.json"
        if metrics_config_file.exists():
            with open(metrics_config_file, "r") as f:
                self.dataset_metrics_config = json.load(f)
                logger.info(f"Loaded dataset-specific metrics config from {metrics_config_file}")

        # Load samples
        self.dataset_data = self._load_samples()

    def _copy_source_data(self) -> None:
        """Copy data from source directory to task data directory."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Copy all files from source to data directory
        for file_path in self.source_path.glob("*"):
            if file_path.is_file():
                dest_path = self.data_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                logger.debug(f"Copied {file_path.name}")

    def _load_samples(self) -> List[Dict]:
        """Load samples from data directory.
        
        Returns:
            List of sample dicts with image paths, schemas, and expected outputs
        """
        samples = []

        # Load expected outputs
        expected_file = self.data_dir / "expected.json"
        if not expected_file.exists():
            raise FileNotFoundError(f"Expected outputs file not found: {expected_file}")

        with open(expected_file, "r") as f:
            expected_outputs = json.load(f)

        # Create samples for each expected output
        for idx, (file_key, expected) in enumerate(expected_outputs.items()):
            # Look for the image file
            image_path = self.data_dir / file_key
            if not image_path.exists():
                # Try common image extensions
                for ext in [".jpg", ".jpeg", ".png", ".pdf"]:
                    test_path = self.data_dir / f"{file_key}{ext}"
                    if test_path.exists():
                        image_path = test_path
                        break

            if not image_path.exists():
                logger.warning(f"Image file not found for {file_key}, skipping")
                continue

            sample = {
                "id": f"{self.get_task_name()}_{idx}",
                "image_path": str(image_path),
                "schema": self.schema,
                "expected": expected,
                "file_key": file_key,
            }
            samples.append(sample)

        logger.info(f"Loaded {len(samples)} multimodal samples")
        return samples

    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset samples.
        
        Args:
            limit: Maximum number of samples to return
            
        Yields:
            Sample dictionaries with image paths and schemas
        """
        if self.dataset_data is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        count = 0
        for sample in self.dataset_data:
            if limit and count >= limit:
                break
            yield sample
            count += 1

    def get_prompt(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Build prompts for multimodal structured extraction.
        
        Includes schema information in the prompt.
        
        Args:
            sample: Sample with schema
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        schema = sample.get("schema", {})
        schema_str = json.dumps(schema, indent=2) if schema else ""

        # Use custom template if provided
        if hasattr(self, "user_prompt_template") and "{schema}" in self.user_prompt_template:
            user_prompt = self.user_prompt_template.format(schema=schema_str)
        else:
            # Default format for image extraction
            user_prompt = (
                f"Extract the data from this image following the provided schema.\n"
                f"Return valid JSON matching the schema exactly.\n\n"
                f"Schema:\n{schema_str}"
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
        
        Uses the document extraction metrics calculator.
        
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
        """Aggregate structured extraction metrics across all samples.
        
        Args:
            all_metrics: List of per-sample metrics
            
        Returns:
            Aggregated metrics dict
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
            "document_extraction_score",
            "extraction_quality_score",
            "strict_type_compliance_rate",
        ]

        for metric_name in numeric_metrics:
            values = [m.get(metric_name, 0) for m in all_metrics if m.get("valid", False)]
            if values:
                aggregated[metric_name] = sum(values) / len(values)

        return aggregated

