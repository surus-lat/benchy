"""Image extraction task implementation.

Evaluates model capabilities to extract structured data from images
following a JSON schema.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Data directory relative to this module
DATA_DIR = Path(__file__).parent / ".data"


class ImageExtractionTask:
    """Task for extracting structured data from images.
    
    Implements the BaseTask protocol for the generic benchmark runner.
    """
    
    TASK_NAME: str = "image_extraction"

    def __init__(self, config: Dict):
        """Initialize the image extraction task.

        Args:
            config: Configuration dictionary with:
                - source_dir: Path to source data (images, schema, expected)
                - prompts.system: System prompt
                - prompts.user: User prompt template
        """
        self.config = config
        self.source_dir = Path(config.get("source_dir", ""))
        self.data_dir = DATA_DIR
        self.dataset: Optional[List[Dict]] = None
        self.schema: Optional[Dict] = None
        
        # Lazy init metrics calculator
        self._metrics_calc = None
    
    @property
    def metrics_calculator(self):
        """Lazy initialization of metrics calculator from structured extraction."""
        if self._metrics_calc is None:
            from ..structured.metrics import MetricsCalculator
            self._metrics_calc = MetricsCalculator(self.config)
        return self._metrics_calc

    def load(self) -> None:
        """Load dataset, copying from source if needed."""
        # Check if data exists, otherwise copy from source
        if not (self.data_dir / "schema.json").exists():
            if not self.source_dir.exists():
                raise FileNotFoundError(
                    f"Source directory not found: {self.source_dir}\n"
                    f"Please provide a valid source_dir in the task config."
                )
            logger.info(f"Copying data from {self.source_dir} to {self.data_dir}")
            self._copy_source_data()
        
        # Load schema
        schema_file = self.data_dir / "schema.json"
        logger.info(f"Loading schema from {schema_file}")
        with open(schema_file, "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        
        # Load expected data
        datos_file = self.data_dir / "datos.json"
        logger.info(f"Loading expected data from {datos_file}")
        with open(datos_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        # Build dataset with image paths
        self.dataset = []
        jpgs_dir = self.data_dir / "jpgs"
        
        for idx, item in enumerate(raw_data):
            filename = item.get("filename", f"image_{idx}")
            image_path = jpgs_dir / f"{filename}.jpg"
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}, skipping")
                continue
            
            # Remove filename from expected (not part of extraction)
            expected = {k: v for k, v in item.items() if k != "filename"}
            
            self.dataset.append({
                "id": f"img_{idx:04d}_{filename}",
                "image_path": str(image_path),
                "schema": self.schema,
                "expected": expected,
                "filename": filename,
            })
        
        logger.info(f"Loaded {len(self.dataset)} samples")
    
    def _copy_source_data(self) -> None:
        """Copy source data to local .data directory."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy schema.json
        src_schema = self.source_dir / "schema.json"
        if src_schema.exists():
            shutil.copy2(src_schema, self.data_dir / "schema.json")
        else:
            raise FileNotFoundError(f"schema.json not found in {self.source_dir}")
        
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
        
        logger.info(f"Copied source data to {self.data_dir}")

    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict]:
        """Iterate over dataset samples.

        Args:
            limit: Maximum number of samples to return (None for all)

        Yields:
            Dictionary with sample data including image_path
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        data = self.dataset
        if limit is not None:
            data = self.dataset[:min(limit, len(self.dataset))]
            logger.info(f"Limited to {len(data)} samples")

        for sample in data:
            yield sample

    def get_prompt(self, sample: Dict) -> Tuple[str, str]:
        """Build prompt messages for a sample.

        Args:
            sample: Sample dictionary with image_path

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.config.get("prompts", {}).get(
            "system",
            "You are an expert at extracting structured data from images."
        )
        user_template = self.config.get("prompts", {}).get(
            "user",
            "Extract data from this image following the schema."
        )
        
        # Format user prompt with schema
        schema_str = json.dumps(sample["schema"], indent=2)
        user_prompt = f"{user_template}\n\nSchema:\n{schema_str}"

        return system_prompt, user_prompt

    def get_task_name(self) -> str:
        """Get the task identifier."""
        return self.TASK_NAME

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict,
    ) -> Dict[str, Any]:
        """Calculate metrics for a single prediction.
        
        Uses MetricsCalculator from structured extraction for consistency.
        
        Args:
            prediction: Model output
            expected: Expected output
            sample: Full sample dict (contains schema)
            
        Returns:
            Metrics dictionary
        """
        return self.metrics_calculator.calculate_all(
            prediction=prediction,
            expected=expected,
            schema=sample.get("schema", {}),
            error=None if prediction is not None else "No prediction",
        )

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate per-sample metrics into summary statistics.
        
        Args:
            all_metrics: List of per-sample metric dictionaries
            
        Returns:
            Aggregated metrics
        """
        return self.metrics_calculator.aggregate_metrics(all_metrics)

    @property
    def is_multimodal(self) -> bool:
        """Image extraction requires multimodal (vision) support."""
        return True
    
    @property
    def requires_schema(self) -> bool:
        """Image extraction uses JSON schemas."""
        return True

