"""Paraloq JSON Data Extraction Task."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any

from ..base import StructuredExtractionTaskBase

logger = logging.getLogger(__name__)


class ParaloqTask(StructuredExtractionTaskBase):
    """Task for paraloq/json_data_extraction dataset.
    
    Implements the BaseTask protocol for the generic benchmark runner.
    """

    def __init__(self, config: Dict):
        """Initialize the Paraloq task.

        Args:
            config: Configuration dictionary with dataset settings
        """
        super().__init__(config)
        self.data_file = Path(config["dataset"]["data_file"])
        self.dataset = None

    def load(self) -> None:
        """Load the dataset from local JSONL file."""
        logger.info(f"Loading dataset from {self.data_file}")
        
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.data_file}\n"
                "Please run: python download_dataset.py"
            )
        
        self.dataset = []
        with open(self.data_file, "r") as f:
            for line in f:
                self.dataset.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.dataset)} samples")

    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict]:
        """Iterate over dataset samples.

        Args:
            limit: Maximum number of samples to return (None for all)

        Yields:
            Dictionary with sample data
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        dataset_to_use = self.dataset
        if limit is not None:
            dataset_to_use = self.dataset[:min(limit, len(self.dataset))]
            logger.info(f"Limited to {len(dataset_to_use)} samples")

        for sample in dataset_to_use:
            # Data is already preprocessed from JSONL
            yield sample

    def get_prompt(self, sample: Dict) -> tuple[str, str]:
        """Build prompt messages for a sample.

        Args:
            sample: Sample dictionary

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.config["prompts"]["system"]
        user_template = self.config["prompts"]["user"]

        user_prompt = user_template.format(
            text=sample["text"],
            schema=json.dumps(sample["schema"], indent=2)
        )

        return system_prompt, user_prompt

    def get_task_name(self) -> str:
        """Get the task identifier."""
        return "paraloq"

