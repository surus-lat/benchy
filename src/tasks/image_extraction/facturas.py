"""Facturas - invoice/receipt structured data extraction from images."""

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
        """
        Initialize the Facturas task, setting up data source and loading dataset metrics.
        
        Parameters:
            config (Optional[Dict[str, Any]]): Optional configuration dictionary. If it contains a "source_dir" key, that path is used as the dataset source directory; otherwise a default test data directory is used. The initializer calls the superclass constructor and then loads any dataset-specific metrics configuration into self.metrics_config.
        """
        # Set source_dir BEFORE calling super().__init__()
        if config and "source_dir" in config:
            self.source_dir = Path(config["source_dir"])
        elif self.source_dir is None:
            # Default to test data directory
            self.source_dir = Path(__file__).parent.parent.parent.parent / "test_image_request"

        super().__init__(config)

        # Load dataset-specific metrics config if available
        self._load_dataset_metrics_config()

    def _load_dataset_metrics_config(self):
        """
        Load dataset-specific metrics configuration from the task data directory and merge it into the runtime metrics configuration.
        
        If a metrics_config.json file exists in the data directory, its contents are parsed and merged into self.metrics_config, overriding or extending existing keys as needed. Any errors during loading or parsing are logged as warnings and do not raise exceptions.
        """
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
        """
        Recursively merge keys from `override` into `base`, modifying `base` in place.
        
        When a key exists in both dictionaries and both corresponding values are dictionaries,
        their contents are merged recursively. For all other cases, the value from `override`
        replaces the value in `base`.
        
        Parameters:
            base (Dict): Destination dictionary that will be updated in place.
            override (Dict): Source dictionary whose keys and values override or extend `base`.
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _load_samples(self) -> list:
        """
        Load dataset samples from the dataset's datos.json file and assemble sample records with resolved image paths.
        
        If datos.json is missing in self.data_dir, attempts to copy dataset files from self.source_dir (raises FileNotFoundError if source_dir is not set or does not exist). Optionally loads schema.json when present; each returned sample will include the schema when available. Each sample is a dict containing: id, image_path, expected (input item with "filename" and "cae" removed), and filename. Items whose referenced image file is missing are skipped.
        
        Returns:
            list: A list of sample dictionaries ready for processing.
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
            filename = item.get("filename", f"image_{idx}")
            image_path = jpgs_dir / f"{filename}.jpg"

            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}, skipping")
                continue

            # Remove filename and cae from expected (not part of extraction)
            expected = {k: v for k, v in item.items() if k not in ("filename", "cae")}

            sample = {
                "id": f"factura_{idx:04d}_{filename}",
                "image_path": str(image_path),
                "expected": expected,
                "filename": filename,
            }

            # Add schema only if it exists
            if schema:
                sample["schema"] = schema

            dataset.append(sample)

        logger.info(f"Loaded {len(dataset)} samples")
        return dataset

    def _copy_source_data(self) -> None:
        """
        Copy dataset files from the configured source_dir into the task's local data directory.
        
        Creates self.data_dir if needed and copies the following from self.source_dir:
        - Optional schema.json (logged if present; absence is allowed for schema-less mode).
        - Required datos.json (must exist).
        - Required jpgs directory (copied; any existing destination jpgs directory is replaced).
        - Optional metrics_config.json (logged if present).
        
        Raises:
            FileNotFoundError: If datos.json or the jpgs directory is missing in self.source_dir.
        """
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
            logger.info(f"Copied metrics_config.json from source")

        logger.info(f"Copied source data to {self.data_dir}")

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Builds the user-facing prompt for a sample, embedding the sample schema when present.
        
        Parameters:
            sample (Dict[str, Any]): Sample dictionary. If it contains a "schema" key, the schema (a JSON-serializable object) will be included in the prompt.
        
        Returns:
            str: The formatted prompt. If `sample` contains a schema, the prompt includes a pretty-printed JSON schema block; otherwise it returns a minimal instruction for schema-free extraction.
        """
        # If schema is present, include it in the prompt
        if "schema" in sample:
            schema_str = json.dumps(sample["schema"], indent=2)
            return f"{self.user_prompt_template}\n\nSchema:\n{schema_str}"
        else:
            # No schema mode (SURUS Factura)
            return "Extract all data from this invoice/receipt image as structured JSON."
