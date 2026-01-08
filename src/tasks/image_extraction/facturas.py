"""Facturas - invoice/receipt structured data extraction from images."""

import copy
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from ..common import MultimodalStructuredHandler

logger = logging.getLogger(__name__)


class Facturas(MultimodalStructuredHandler):
    """Extract structured data from invoice/factura images.
    
    This task evaluates vision-language models on structured data extraction
    from invoice and receipt images. It uses lenient metrics optimized for
    document extraction where numeric accuracy is critical.
    
    Compatible with:
    - SURUS Factura endpoint (no schema needed, returns structured JSON)
    - Regular vision models with schema support (OpenAI, Anthropic, etc.)
    """

    # Task configuration
    name = "facturas"
    display_name = "Facturas"
    description = "Extract structured data from invoice/factura images"

    # Multimodal configuration
    input_type = "image"
    requires_multimodal = True
    requires_files = True
    requires_schema = False  # SURUS Factura doesn't need schema, returns structured JSON

    # Source directory (set by config or use default test data)
    source_dir: Optional[Path] = None

    # Prompts (optional - some endpoints like SURUS Factura ignore prompts)
    system_prompt = (
        "You are an expert in extracting information from invoices and receipts. "
        "Extract all the required fields following the required schema."
    )

    user_prompt_template = (
        "Extract the data from this invoice/receipt image following the provided schema.\n"
        "Return valid JSON matching the schema exactly."
    )

    # Lenient metrics for document extraction
    metrics_config = {
        "partial_matching": {
            "string": {
                "exact_threshold": 0.85,  # More lenient than default 0.95
                "partial_threshold": 0.40,  # More lenient than default 0.50
            }
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize facturas task."""
        # Set source_dir BEFORE calling super().__init__()
        if config and "source_dir" in config:
            self.source_dir = Path(config["source_dir"])
        elif self.source_dir is None:
            self.source_dir = None

        self.metrics_config = copy.deepcopy(self.metrics_config)

        super().__init__(config)

        # Load dataset-specific metrics config if available
        self._load_dataset_metrics_config()

    def _load_dataset_metrics_config(self):
        """Load dataset-specific metrics configuration from .data/metrics_config.json."""
        metrics_config_file = self.data_dir / "metrics_config.json"
        if metrics_config_file.exists():
            logger.info(f"Loading dataset metrics config from {metrics_config_file}")
            try:
                with open(metrics_config_file, "r", encoding="utf-8") as f:
                    dataset_config = json.load(f)

                # Deep merge dataset config into task metrics_config
                self._deep_merge(self.metrics_config, dataset_config)
                logger.info(f"Merged dataset metrics config: {list(dataset_config.keys())}")
            except Exception as e:
                logger.warning(f"Failed to load dataset metrics config: {e}")

    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """Recursively merge override dict into base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _load_samples(self) -> list:
        """Load facturas dataset samples.
        
        Overrides MultimodalStructuredHandler._load_samples() to use
        datos.json instead of expected.json.

        Expected structure:
            .data/
            ├── schema.json          # JSON Schema (optional)
            ├── datos.json           # Ground truth array
            ├── metrics_config.json  # Optional dataset-specific config
            └── jpgs/                # Invoice/receipt images
                ├── factura_001.jpg
                └── ...
        """
        # Check if data exists, otherwise copy from source
        if not (self.data_dir / "datos.json").exists():
            if not self.source_dir or not self.source_dir.exists():
                raise FileNotFoundError(
                    f"Source directory not found: {self.source_dir}\n"
                    f"Please provide a valid source_dir in the task config."
                )
            logger.info(f"Copying data from {self.source_dir} to {self.data_dir}")
            self._copy_source_data()

        # Reload metrics config after data is copied
        self._load_dataset_metrics_config()

        # Load schema (optional for this task)
        schema = None
        schema_file = self.data_dir / "schema.json"
        if schema_file.exists():
            logger.info(f"Loading schema from {schema_file}")
            with open(schema_file, "r", encoding="utf-8") as f:
                schema = json.load(f)
        else:
            logger.info("No schema.json found - running without schema (SURUS Factura mode)")

        # Load expected data
        datos_file = self.data_dir / "datos.json"
        logger.info(f"Loading expected data from {datos_file}")
        with open(datos_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Build dataset with image paths
        dataset = []
        jpgs_dir = self.data_dir / "jpgs"

        for idx, item in enumerate(raw_data):
            raw_name = item.get("filename")
            if raw_name:
                clean_name = Path(str(raw_name)).stem
            else:
                clean_name = f"image_{idx}"
            image_path = jpgs_dir / f"{clean_name}.jpg"

            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}, skipping")
                continue

            # Remove filename and cae from expected (not part of extraction)
            expected = {k: v for k, v in item.items() if k not in ("filename", "cae")}

            sample = {
                "id": f"factura_{idx:04d}_{clean_name}",
                "image_path": str(image_path),
                "expected": expected,
                "filename": raw_name or clean_name,
            }

            # Add schema only if it exists
            if schema:
                sample["schema"] = schema

            dataset.append(sample)

        logger.info(f"Loaded {len(dataset)} samples")
        return dataset

    def _copy_source_data(self) -> None:
        """Copy source data to local .data directory."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Copy schema.json (optional)
        src_schema = self.source_dir / "schema.json"
        if src_schema.exists():
            shutil.copy2(src_schema, self.data_dir / "schema.json")
            logger.info("Copied schema.json from source")
        else:
            logger.info("No schema.json in source (SURUS Factura mode - schema not required)")

        # Copy datos.json
        src_datos = self.source_dir / "datos.json"
        if src_datos.exists():
            shutil.copy2(src_datos, self.data_dir / "datos.json")
        else:
            raise FileNotFoundError(f"datos.json not found in {self.source_dir}")

        # Copy jpgs directory
        src_jpgs = self.source_dir / "jpgs"
        dst_jpgs = self.data_dir / "jpgs"
        if src_jpgs.exists():
            if dst_jpgs.exists():
                shutil.rmtree(dst_jpgs)
            shutil.copytree(src_jpgs, dst_jpgs)
        else:
            raise FileNotFoundError(f"jpgs directory not found in {self.source_dir}")

        # Copy metrics_config.json if it exists (optional)
        src_metrics_config = self.source_dir / "metrics_config.json"
        if src_metrics_config.exists():
            shutil.copy2(src_metrics_config, self.data_dir / "metrics_config.json")
            logger.info("Copied metrics_config.json from source")

        logger.info(f"Copied source data to {self.data_dir}")

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format the user prompt for a sample."""
        # If schema is present, include it in the prompt
        if "schema" in sample:
            schema_str = json.dumps(sample["schema"], indent=2)
            return f"{self.user_prompt_template}\n\nSchema:\n{schema_str}"
        else:
            # No schema mode (SURUS Factura)
            return "Extract all data from this invoice/receipt image as structured JSON."
