"""Template task implementation.

To create a new task:
1. Copy this folder: cp -r src/tasks/_template src/tasks/my_task
2. Edit this file:
   - Set DATASET_NAME to your HuggingFace dataset
   - Set TASK_NAME to your task identifier
   - Override preprocess_sample() to transform HF samples to eval format
   - Override calculate_metrics() for task-specific scoring
3. Edit run.py: rename run_template_task to run_my_task
4. Add task to pipeline.py dispatch
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Any, List, Tuple

from ...engine.protocols import BaseTask
from ...common import download_huggingface_dataset, save_to_jsonl

logger = logging.getLogger(__name__)

# Data directory relative to this module
DATA_DIR = Path(__file__).parent / ".data"


class TemplateTask(BaseTask):
    """Template task - copy and customize for your task.
    
    Inherits from BaseTask protocol for IDE autocompletion and type checking.
    """
    
    # === CONFIGURE THESE ===
    DATASET_NAME: str = "your-org/your-dataset"  # HuggingFace dataset identifier
    TASK_NAME: str = "template_task"  # Task identifier for logging/checkpointing
    DATASET_SPLIT: str = "train"  # HuggingFace dataset split
    DATA_FILE: str = "data.jsonl"  # Processed data filename

    def __init__(self, config: Dict):
        """Initialize the task.
        
        Args:
            config: Configuration dictionary with:
                - dataset.data_file: (optional) Override data file path
                - prompts.system: System prompt template
                - prompts.user: User prompt template (with {placeholders})
        """
        self.config = config
        
        # Use config path if provided, otherwise use default
        data_file = config.get("dataset", {}).get("data_file")
        if data_file:
            self.data_file = Path(data_file)
        else:
            self.data_file = DATA_DIR / self.DATA_FILE
        
        self.dataset: Optional[List[Dict]] = None

    def load(self) -> None:
        """Load dataset, auto-downloading from HuggingFace if needed."""
        # Auto-download if data file doesn't exist
        if not self.data_file.exists():
            logger.info(f"Data file not found: {self.data_file}")
            logger.info(f"Downloading from HuggingFace: {self.DATASET_NAME}")
            self._download_and_preprocess()
        
        # Load from JSONL
        logger.info(f"Loading dataset from {self.data_file}")
        self.dataset = []
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.dataset.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.dataset)} samples")

    def _download_and_preprocess(self) -> None:
        """Download from HuggingFace and preprocess samples."""
        # Download raw dataset
        raw_samples = download_huggingface_dataset(
            dataset_name=self.DATASET_NAME,
            split=self.DATASET_SPLIT,
            cache_dir=str(DATA_DIR / "cache"),
        )
        
        # Preprocess each sample
        processed = []
        skipped = 0
        
        for idx, raw_sample in enumerate(raw_samples):
            result = self.preprocess_sample(raw_sample, idx)
            if result is not None:
                # Ensure ID is set
                if "id" not in result:
                    result["id"] = f"sample_{len(processed)}"
                processed.append(result)
            else:
                skipped += 1
        
        logger.info(f"Processed {len(processed)} samples, skipped {skipped}")
        
        # Save to JSONL
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        save_to_jsonl(processed, self.data_file)

    def preprocess_sample(self, raw_sample: Dict, idx: int) -> Optional[Dict]:
        """Transform a HuggingFace sample to eval format.
        
        Override this method to customize how raw HF samples are converted
        to the evaluation format. Return None to skip a sample.
        
        Args:
            raw_sample: Raw sample dictionary from HuggingFace dataset
            idx: Sample index (for generating IDs)
            
        Returns:
            Processed sample dict with at minimum:
            - id: Unique sample identifier
            - text: Input text for the model
            - expected: Expected output for metrics calculation
            
            Return None to skip this sample.
        """
        # Default: simple pass-through with common field names
        # Override this for your specific dataset format
        return {
            "id": f"sample_{idx}",
            "text": raw_sample.get("text", raw_sample.get("input", "")),
            "expected": raw_sample.get("expected", raw_sample.get("label", raw_sample.get("output"))),
        }

    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict]:
        """Iterate over dataset samples.
        
        Each sample has at minimum:
        - id: Unique sample identifier
        - text: Input text
        - expected: Expected output for metrics
        
        Args:
            limit: Maximum samples to return (None for all)
            
        Yields:
            Sample dictionaries
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
        """Build prompts for a sample.
        
        Used by LLM interfaces to construct chat messages.
        Override this for custom prompt formatting.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.config.get("prompts", {}).get(
            "system", 
            "You are a helpful assistant."
        )
        user_template = self.config.get("prompts", {}).get(
            "user",
            "{text}"
        )

        # Format user prompt with sample data
        user_prompt = user_template.format(
            text=sample.get("text", ""),
            # Add more placeholders as needed in your override
        )

        return system_prompt, user_prompt

    def get_task_name(self) -> str:
        """Return task identifier for logging and checkpointing."""
        return self.TASK_NAME

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict,
    ) -> Dict[str, Any]:
        """Calculate metrics for a single prediction.
        
        Override this for task-specific metrics.
        
        Args:
            prediction: Model output (parsed if applicable)
            expected: Expected output from sample
            sample: Full sample dict for additional context
            
        Returns:
            Dictionary of metric names to values.
            Must include 'valid' (bool) to indicate if prediction was usable.
        """
        if prediction is None:
            return {
                "valid": False,
                "score": 0.0,
                "error": "No prediction",
            }
        
        # Default: simple exact match
        # Override for more sophisticated metrics
        exact_match = prediction == expected
        
        return {
            "valid": True,
            "exact_match": exact_match,
            "score": 1.0 if exact_match else 0.0,
        }

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate per-sample metrics into summary statistics.
        
        Override this if you need custom aggregation logic.
        
        Args:
            all_metrics: List of per-sample metric dicts from calculate_metrics()
            
        Returns:
            Aggregated summary metrics
        """
        if not all_metrics:
            return {
                "total_samples": 0,
                "valid_samples": 0,
                "score": 0.0,
            }
        
        valid = [m for m in all_metrics if m.get("valid")]
        
        return {
            "total_samples": len(all_metrics),
            "valid_samples": len(valid),
            "exact_match_rate": sum(m.get("exact_match", False) for m in valid) / len(valid) if valid else 0,
            "score": sum(m.get("score", 0) for m in valid) / len(valid) if valid else 0,
            "error_count": len(all_metrics) - len(valid),
        }

    # Capability flags - override as needed
    @property
    def is_multimodal(self) -> bool:
        """Whether this task requires images/audio."""
        return False
    
    @property
    def requires_schema(self) -> bool:
        """Whether samples include JSON schema for structured output."""
        return False
