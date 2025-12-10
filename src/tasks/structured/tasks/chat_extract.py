"""Chat Structured Extraction Task."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Any, List

from ..base import StructuredExtractionTaskBase

logger = logging.getLogger(__name__)


class ChatExtractTask(StructuredExtractionTaskBase):
    """Task for mauroibz/chat_structured_extraction dataset.
    
    Implements the BaseTask protocol for the generic benchmark runner.
    """

    def __init__(self, config: Dict):
        """Initialize the ChatExtract task.

        Args:
            config: Configuration dictionary with dataset settings
        """
        super().__init__(config)
        self.data_file = Path(config["dataset"]["data_file"])
        self.schema_file = Path(config["dataset"].get("schema_file", ""))
        self.dataset = None
        self.schema = None

    def load(self) -> None:
        """Load the dataset and schema from local files."""
        logger.info(f"Loading dataset from {self.data_file}")
        
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.data_file}\n"
                "Please ensure the dataset has been downloaded."
            )
        
        # Load dataset
        self.dataset = []
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line in f:
                self.dataset.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.dataset)} samples")
        
        # Note: Schema is embedded in each preprocessed sample, so we don't need to load it separately
        # But we keep this for compatibility and potential future use
        if self.schema_file and self.schema_file.exists():
            logger.debug(f"Schema file found at {self.schema_file} (schema is in samples)")

    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict]:
        """Iterate over dataset samples.

        Args:
            limit: Maximum number of samples to return (None for all)

        Yields:
            Dictionary with sample data (already in benchmark format from preprocessing)
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        dataset_to_use = self.dataset
        if limit is not None:
            dataset_to_use = self.dataset[:min(limit, len(self.dataset))]
            logger.info(f"Limited to {len(dataset_to_use)} samples")

        for sample in dataset_to_use:
            # Data is already preprocessed from JSONL by download_and_preprocess_chat_extraction
            # which formats it in the same structure as paraloq samples
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
        return "chat_extract"
