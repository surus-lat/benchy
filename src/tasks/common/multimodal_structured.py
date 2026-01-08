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
        """
        Constructs a handler configured to load multimodal data and metrics from a required source directory.
        
        Parameters:
            config (Optional[Dict[str, Any]]): Optional configuration overrides. If present, the key `"source_dir"` will override the class-level `source_dir` used to locate image/file sources and dataset files.
        
        Raises:
            ValueError: If no `source_dir` is configured after applying `config`.
        """
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
        """
        Lazily initialize and return the metrics calculator configured for this handler.
        
        Returns:
            MetricsCalculator: An instance configured to compute document-extraction and schema-based metrics, cached after first creation.
        
        Raises:
            ImportError: If the MetricsCalculator implementation cannot be imported.
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
        """
        Produce a configuration dictionary that merges the handler's base config with class-level and dataset-specific metrics settings.
        
        The returned dictionary is a deep copy of self.config (or an empty dict) with a "metrics" mapping created if needed and updated by self.metrics_config and then by self.dataset_metrics_config. Values from dataset-specific configuration override or extend class-level metrics entries.
        
        Returns:
            merged (Dict): The merged configuration dictionary.
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
        """
        Recursively merge values from `override` into `base`, modifying `base` in place.
        
        Parameters:
            base (Dict): Target dictionary to be updated (modified in place).
            override (Dict): Source dictionary whose values overwrite or are merged into `base`; when both `base[key]` and `override[key]` are dicts, the merge recurses.
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def load(self) -> None:
        """
        Load dataset artifacts into the handler, copying files from the configured source directory when necessary.
        
        This method ensures a schema is present in the task data directory (copying files from self.source_dir if the schema is missing), then loads:
        - schema.json into self.schema (if present),
        - metrics_config.json into self.dataset_metrics_config (if present),
        - expected dataset samples via self._load_samples() into self.dataset_data.
        
        Raises:
            FileNotFoundError: if the configured source directory does not exist when a copy is required.
        """
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
        """
        Load dataset samples by pairing expected outputs with corresponding files in the task data directory.
        
        Searches for each entry in `data_dir/expected.json` and resolves an input file by using the entry key or the key plus common extensions (.jpg, .jpeg, .png, .pdf). Entries without a resolvable file are skipped.
        
        Returns:
            A list of sample dictionaries, each containing:
              - `id` (str): unique sample identifier,
              - `image_path` (str): path to the input file,
              - `schema` (dict|None): task schema,
              - `expected` (Any): expected output for the sample,
              - `file_key` (str): original key from expected.json.
        
        Raises:
            FileNotFoundError: if `data_dir/expected.json` does not exist.
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
        """
        Yield dataset samples in order, optionally limiting the number returned.
        
        Parameters:
            limit (Optional[int]): Maximum number of samples to yield; if None, yield all samples.
        
        Returns:
            Iterator[Dict[str, Any]]: An iterator over sample dictionaries. Each sample contains keys such as the image path and the associated schema.
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
        """
        Constructs the system and user prompts for a sample, embedding the sample's schema into the user prompt.
        
        If the handler has a `user_prompt_template` containing the placeholder `{schema}`, that template is used with the sample schema substituted; otherwise a default user prompt instructing extraction and to return JSON matching the schema is produced.
        
        Parameters:
            sample (Dict[str, Any]): Sample dictionary; the optional `"schema"` key is pretty-printed and included in the user prompt.
        
        Returns:
            Tuple[str, str]: `(system_prompt, user_prompt)` where `user_prompt` includes the sample schema as formatted JSON.
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
        """
        Compute structured-extraction metrics for a prediction against the expected output using the handler's metrics calculator.
        
        Parameters:
            prediction: Model output to evaluate (typically a dict or parsed JSON).
            expected: Ground-truth structured output to compare against.
            sample: Sample dictionary; the schema is read from sample.get("schema", {}).
            error: Optional error message observed while producing the prediction.
            error_type: Optional categorical error type.
        
        Returns:
            dict: A dictionary of computed metrics (per-sample structured extraction and validation scores).
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
        Compute metrics representing a failed prediction using the provided error information.
        
        Parameters:
            error (str): Error message describing the failure.
            error_type (Optional[str]): Optional error classification or code.
        
        Returns:
            Dict[str, Any]: Metrics dictionary following the same structure as successful prediction metrics, populated for an error case (no prediction).
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
        Aggregate per-sample structured-extraction metrics into dataset-level statistics.
        
        Parameters:
            all_metrics (List[Dict]): List of per-sample metric dictionaries.
        
        Returns:
            Dict[str, Any]: Aggregated metrics including:
                - total_samples: total number of samples processed
                - valid_samples: number of samples marked as valid
                - error_count: number of invalid samples
                - schema_validity_rate: fraction of samples with schema_validity > 0.5
                - exact_match_rate: fraction of samples with exact_match == True
                - error_rate: fraction of invalid samples
                - <numeric metric names>: averaged numeric scores for each metric present
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
