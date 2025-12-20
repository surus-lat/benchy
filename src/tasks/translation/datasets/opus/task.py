"""OPUS-100 translation task implementation."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Any

from ...base import TranslationTaskBase

logger = logging.getLogger(__name__)


class OpusTask(TranslationTaskBase):
    """Task for OPUS-100 translation evaluation.
    
    Supports English-Spanish and English-Portuguese language pairs.
    """
    
    def __init__(self, config: Dict):
        """Initialize the OPUS task.
        
        Args:
            config: Configuration dictionary with:
                - dataset: dict with data_file path
                - prompts: dict with system and user prompts
                - language_pair: language pair identifier (e.g., "en-es")
        """
        super().__init__(config)
        self.data_file = Path(config["dataset"]["data_file"])
        self.language_pair = config.get("language_pair", "en-es")
        self.dataset = None
    
    def load(self) -> None:
        """Load dataset from JSONL file."""
        logger.info(f"Loading OPUS dataset from {self.data_file}")
        
        if not self.data_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.data_file}\n"
                "Please run the download script first."
            )
        
        self.dataset = []
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line in f:
                self.dataset.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.dataset)} samples for {self.language_pair}")
    
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
            sample: Sample dictionary with source_text, target_text, source_lang, target_lang
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.config["prompts"]["system"]
        user_template = self.config["prompts"]["user"]
        
        # Map language codes to display names
        lang_names = {
            "en": "English",
            "es": "Spanish",
            "pt": "Portuguese",
        }
        
        source_lang = lang_names.get(sample.get("source_lang", "en"), sample.get("source_lang", "en"))
        target_lang = lang_names.get(sample.get("target_lang", "es"), sample.get("target_lang", "es"))
        
        user_prompt = user_template.format(
            source_language=source_lang,
            target_language=target_lang,
            source_text=sample["source_text"],
        )
        
        return system_prompt, user_prompt
    
    def get_task_name(self) -> str:
        """Get the task identifier."""
        return f"opus_{self.language_pair}"
    
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

