"""Multimodal structured output format handler.

This handler combines multimodal input handling (images, files) with structured
JSON output evaluation. Used for tasks like image extraction, document analysis, etc.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

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
        source_dir: Optional source directory path for files.
            If provided, missing datasets can be populated by copying from source_dir.
            If not provided, tasks may ship a pre-populated `.data/` folder or override
            `_copy_source_data()` to download/populate data on demand.
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
    source_dir: Optional[Any] = None
    metrics_config: Optional[Dict[str, Any]] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the multimodal structured handler."""
        super().__init__(config)

        # Optional override source_dir from config if provided.
        # Handler runner passes dataset config under config["dataset"].
        source_dir = None
        if config:
            if "source_dir" in config:
                source_dir = config["source_dir"]
            else:
                dataset_cfg = config.get("dataset")
                if isinstance(dataset_cfg, dict) and dataset_cfg.get("source_dir"):
                    source_dir = dataset_cfg.get("source_dir")
        if source_dir is not None:
            self.source_dir = source_dir

        self.source_path: Optional[Path] = Path(str(self.source_dir)) if self.source_dir else None
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
                merged = self._get_merged_config()
                metrics_cfg = merged.get("metrics", {}) if isinstance(merged, dict) else {}
                strict = bool(metrics_cfg.get("strict", False))
                self._metrics_calc = MetricsCalculator({"metrics": metrics_cfg}, strict=strict)
            except ImportError:
                logger.exception("Could not import MetricsCalculator")
                raise

        return self._metrics_calc

    def _get_merged_config(self) -> Dict:
        """Get merged configuration with dataset-specific metrics config.
        
        Returns:
            Merged configuration dictionary
        """
        import copy

        merged = copy.deepcopy(self.config) if self.config else {}

        # Merge metrics with correct precedence:
        # defaults (class-level) < handler config < dataset-specific metrics config
        if "metrics" not in merged or not isinstance(merged.get("metrics"), dict):
            merged["metrics"] = {}

        if self.metrics_config:
            class_defaults = copy.deepcopy(self.metrics_config)
            self._deep_merge(class_defaults, merged["metrics"])
            merged["metrics"] = class_defaults

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
            try:
                if self.source_path and self.source_path.exists():
                    logger.info(f"Copying data from {self.source_path} to {self.data_dir}")
                else:
                    logger.info(f"Populating dataset in {self.data_dir}")
                self._copy_source_data()
            except Exception as exc:
                if isinstance(exc, FileNotFoundError):
                    raise
                if not self.source_path:
                    raise FileNotFoundError(
                        f"Dataset not found in {self.data_dir}.\n"
                        "Provide a valid dataset source via config['dataset']['source_dir'] "
                        "(or config['source_dir']), ship a pre-populated `.data/`, "
                        "or override _copy_source_data() to download/populate data."
                    ) from exc
                raise

        # Load schema
        schema_file = self.data_dir / "schema.json"
        if schema_file.exists():
            with open(schema_file, "r") as f:
                self.schema = json.load(f)

        # Validate schema was loaded when required
        if self.requires_schema and self.schema is None:
            raise FileNotFoundError(
                f"Schema file not found in {self.data_dir} and requires_schema is True"
            )

        # Load dataset metrics config if available
        metrics_config_file = self.data_dir / "metrics_config.json"
        if metrics_config_file.exists():
            with open(metrics_config_file, "r") as f:
                self.dataset_metrics_config = json.load(f)
                logger.info(f"Loaded dataset-specific metrics config from {metrics_config_file}")

        # Load samples
        self.dataset_data = self._load_samples()

    def _copy_source_data(self) -> None:
        """Populate data from local source_dir or HuggingFace dataset snapshot."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if self.source_path:
            if not self.source_path.exists():
                raise FileNotFoundError(f"source_dir not found: {self.source_path}")

            # Copy all files/directories from source to data directory.
            for entry in self.source_path.iterdir():
                dest_path = self.data_dir / entry.name
                if entry.is_dir():
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(entry, dest_path)
                else:
                    shutil.copy2(entry, dest_path)
            return

        dataset_cfg = self._get_dataset_config()
        repo_id = self._resolve_dataset_repo_id(dataset_cfg)
        if not repo_id:
            raise ValueError(
                f"{self.__class__.__name__} has no source_dir or dataset repo configured. "
                "Set config['dataset']['source_dir'] (or config['source_dir']) or provide "
                "config['dataset']['dataset_url']/repo_id in task metadata/config."
            )

        token = self._resolve_hf_token(dataset_cfg)
        revision = dataset_cfg.get("revision")

        logger.info(f"Downloading dataset snapshot: {repo_id}")
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(self.data_dir),
                local_dir_use_symlinks=False,
                token=token,
                revision=str(revision) if revision else None,
            )
        except Exception as exc:
            raise FileNotFoundError(
                f"Failed to download dataset snapshot from '{repo_id}'. "
                "If the dataset is private/gated, set HF_TOKEN (or config['dataset']['token'])."
            ) from exc

    def _get_dataset_config(self) -> Dict[str, Any]:
        cfg = self.config or {}
        dataset_cfg = cfg.get("dataset")
        return dataset_cfg if isinstance(dataset_cfg, dict) else {}

    def _normalize_repo_id(self, value: Any) -> str:
        text = str(value).strip()
        if not text:
            return text

        parsed = urlparse(text)
        if parsed.scheme and parsed.netloc:
            parts = [p for p in parsed.path.split("/") if p]
            # Accept:
            # - https://huggingface.co/datasets/<owner>/<name>
            # - https://huggingface.co/<owner>/<name>
            if len(parts) >= 3 and parts[0] == "datasets":
                return f"{parts[1]}/{parts[2]}"
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            return text

        if text.startswith("datasets/"):
            return text[len("datasets/") :]
        return text

    def _resolve_dataset_repo_id(self, dataset_cfg: Dict[str, Any]) -> Optional[str]:
        cfg = self.config or {}
        candidates = [
            dataset_cfg.get("repo_id"),
            dataset_cfg.get("dataset_repo_id"),
            dataset_cfg.get("dataset_url"),
            dataset_cfg.get("name"),
            dataset_cfg.get("dataset_name"),
            dataset_cfg.get("dataset"),
            cfg.get("dataset_url"),
            cfg.get("repo_id"),
            getattr(self, "dataset_repo_id", None),
        ]
        for candidate in candidates:
            if candidate:
                repo_id = self._normalize_repo_id(candidate)
                if repo_id:
                    return repo_id
        return None

    def _resolve_hf_token(self, dataset_cfg: Dict[str, Any]) -> Optional[str]:
        token = dataset_cfg.get("token") or dataset_cfg.get("auth_token")
        if token:
            return str(token)
        for env_var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            value = os.getenv(env_var)
            if value:
                return value
        return None

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
        """Aggregate structured extraction metrics across all samples."""
        aggregated = self.metrics_calculator.aggregate_metrics(all_metrics)
        # Backwards-compatible aliases for older summary/log consumers.
        if "field_precision_partial" in aggregated and "field_precision" not in aggregated:
            aggregated["field_precision"] = aggregated.get("field_precision_partial", 0.0)
        if "field_recall_partial" in aggregated and "field_recall" not in aggregated:
            aggregated["field_recall"] = aggregated.get("field_recall_partial", 0.0)
        if "type_accuracy" in aggregated and "strict_type_compliance_rate" not in aggregated:
            aggregated["strict_type_compliance_rate"] = aggregated.get("type_accuracy", 0.0)
        return aggregated
