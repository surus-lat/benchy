"""Template task implementation.

This file defines the task class that:
- Loads dataset
- Provides samples
- Builds prompts
- Calculates metrics
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Any, List

logger = logging.getLogger(__name__)


class TemplateTask:
    """Template task - copy and customize for your task.
    
    Implements the BaseTask protocol required by BenchmarkRunner.
    """

    def __init__(self, config: Dict):
        """Initialize the task.
        
        Args:
            config: Configuration dictionary with:
                - dataset.data_file: Path to JSONL data file
                - prompts.system: System prompt template
                - prompts.user: User prompt template (with {placeholders})
        """
        self.config = config
        self.data_file = Path(config["dataset"]["data_file"])
        self.dataset = None

    def load(self) -> None:
        """Load the dataset from JSONL file."""
        logger.info(f"Loading dataset from {self.data_file}")
        
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.data_file}\n"
                "Please run the download script or create the data file."
            )
        
        self.dataset = []
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.dataset.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.dataset)} samples")

    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict]:
        """Iterate over dataset samples.
        
        Each sample should have at minimum:
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

    def get_prompt(self, sample: Dict) -> tuple[str, str]:
        """Build prompts for a sample.
        
        Used by LLM interfaces to construct chat messages.
        HTTP interfaces may ignore this.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.config["prompts"]["system"]
        user_template = self.config["prompts"]["user"]

        # Format user prompt with sample data
        user_prompt = user_template.format(
            text=sample["text"],
            # Add more placeholders as needed
        )

        return system_prompt, user_prompt

    def get_task_name(self) -> str:
        """Return task identifier for logging and checkpointing."""
        return "template_task"  # Change this!

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict,
    ) -> Dict[str, Any]:
        """Calculate metrics for a single prediction.
        
        Args:
            prediction: Model output (parsed if applicable)
            expected: Expected output from sample
            sample: Full sample dict for additional context
            
        Returns:
            Dictionary of metric names to values
        """
        if prediction is None:
            return {
                "valid": False,
                "score": 0.0,
                "error": "No prediction",
            }
        
        # Example: simple exact match
        # Replace with your task-specific metrics
        exact_match = prediction == expected
        
        return {
            "valid": True,
            "exact_match": exact_match,
            "score": 1.0 if exact_match else 0.0,
        }

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate per-sample metrics into summary statistics.
        
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

    # Optional: capability flags
    @property
    def is_multimodal(self) -> bool:
        """Whether this task requires images/audio."""
        return False
    
    @property
    def requires_schema(self) -> bool:
        """Whether samples include JSON schema for structured output."""
        return False

