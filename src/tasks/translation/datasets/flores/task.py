"""FLORES+ translation task implementation."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Any

from ...base import TranslationTaskBase

logger = logging.getLogger(__name__)


class FloresTask(TranslationTaskBase):
    """Task for FLORES+ bidirectional translation evaluation.
    
    Supports multiple bidirectional language pairs (e.g., eng_spa, por_spa).
    """
    
    def __init__(self, config: Dict):
        """Initialize the FLORES task.
        
        Args:
            config: Configuration dictionary with:
                - dataset: dict with data_dir path and language_pair
                - prompts: dict with system and user prompts
                - language_pair: language pair identifier (e.g., "eng_spa")
                - split: dataset split to use ("dev" or "devtest")
        """
        super().__init__(config)
        self.data_dir = Path(config["dataset"]["data_dir"])
        self.language_pair = config.get("language_pair", "eng_spa")
        self.split = config.get("split", "devtest")
        self.dataset = None
    
    def load(self) -> None:
        """Load dataset from JSONL file."""
        data_file = self.data_dir / self.language_pair / f"{self.split}.jsonl"
        
        logger.info(f"Loading FLORES dataset from {data_file}")
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {data_file}\n"
                f"Language pair: {self.language_pair}, Split: {self.split}\n"
                "Please run the download script first."
            )
        
        self.dataset = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                self.dataset.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.dataset)} samples for {self.language_pair} ({self.split})")
    
    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict]:
        """Iterate over dataset samples.
        
        Args:
            limit: Maximum number of samples to return (None for all)
            
        Yields:
            Sample dictionaries with id, source_text, target_text, etc.
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        dataset_to_use = self.dataset
        if limit is not None:
            dataset_to_use = self.dataset[:min(limit, len(self.dataset))]
            logger.info(f"Limited to {len(dataset_to_use)} samples")
        
        for sample in dataset_to_use:
            yield sample
    
    def get_prompt(self, sample: Dict) -> tuple[str, str]:
        """Build prompt messages for a sample.
        
        Args:
            sample: Sample dictionary with source_language, target_language, source_text, target_text
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.config["prompts"]["system"]
        user_template = self.config["prompts"]["user"]
        
        # FLORES samples already have language names
        source_lang = sample.get("source_language", "Unknown")
        target_lang = sample.get("target_language", "Unknown")
        source_text = sample.get("source_text", "")
        
        user_prompt = user_template.format(
            source_language=source_lang,
            target_language=target_lang,
            source_text=source_text,
        )
        
        return system_prompt, user_prompt
    
    def get_task_name(self) -> str:
        """Get the task identifier."""
        return f"flores_{self.language_pair}_{self.split}"
    
    def calculate_metrics(
        self,
        prediction: any,
        expected: any,
        sample: Dict,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for a single prediction.
        
        Uses the base class implementation which delegates to metrics calculator.
        """
        # Expected is the target_text from the sample
        expected_text = sample.get("target_text", expected)
        return super().calculate_metrics(
            prediction=prediction,
            expected=expected_text,
            sample=sample,
            error=error,
            error_type=error_type,
        )

