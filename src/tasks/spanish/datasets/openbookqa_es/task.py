"""OpenBookQA-es task implementation.

Open-domain question answering in Spanish.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Any, Tuple

from ...base import SpanishTaskBase
from .....common.dataset_utils import download_huggingface_dataset, save_to_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / ".data" / "openbookqa_es"


class OpenBookQaEsTask(SpanishTaskBase):
    """OpenBookQA-es task: Open-domain QA in Spanish."""
    
    def __init__(self, config: Dict):
        """Initialize the OpenBookQA-es task."""
        super().__init__(config)
        
        dataset_config = config.get("dataset", {})
        self.dataset_path = dataset_config.get("dataset_path", "BSC-LT/openbookqa-es")
        self.split = dataset_config.get("split", "test")
        
        self.data_file = DATA_DIR / f"{self.split}.jsonl"
        self.dataset: Optional[list] = None
    
    def load(self) -> None:
        """Load dataset, auto-downloading from HuggingFace if needed."""
        if not self.data_file.exists():
            logger.info(f"Data file not found: {self.data_file}")
            logger.info(f"Downloading from HuggingFace: {self.dataset_path}")
            self._download_and_preprocess()
        
        logger.info(f"Loading dataset from {self.data_file}")
        self.dataset = []
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    expected = sample.get("expected")
                    expected_idx = sample.get("expected_idx")
                    if isinstance(expected, str) and expected_idx is not None:
                        sample["expected"] = expected_idx
                    self.dataset.append(sample)
        
        logger.info(f"Loaded {len(self.dataset)} samples")
    
    def _download_and_preprocess(self) -> None:
        """Download from HuggingFace and preprocess samples."""
        raw_samples = download_huggingface_dataset(
            dataset_name=self.dataset_path,
            split=self.split,
            cache_dir=str(DATA_DIR / "cache"),
        )
        
        processed = []
        for idx, raw_sample in enumerate(raw_samples):
            question_stem = raw_sample.get("question_stem", "")
            answer_key = raw_sample.get("answerKey", "").strip()
            
            # Get choices from choices.text
            choices_data = raw_sample.get("choices", {})
            choices_text = choices_data.get("text", [])
            choices_labels = choices_data.get("label", [])
            
            # Find expected index
            try:
                expected_idx = choices_labels.index(answer_key)
            except (ValueError, AttributeError):
                # Fallback: try to match answer_key directly
                expected_idx = 0
                logger.warning(f"Could not find answer_key {answer_key} in choices, using index 0")
            
            processed.append({
                "id": raw_sample.get("id", f"openbookqa_es_{idx}"),
                "text": question_stem,
                "choices": choices_text,
                "expected": expected_idx,
                "expected_idx": expected_idx,
                "raw_sample": raw_sample,
            })
        
        logger.info(f"Processed {len(processed)} samples")
        
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        save_to_jsonl(processed, self.data_file)
    
    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict]:
        """Iterate over dataset samples."""
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        data = self.dataset
        if limit is not None:
            data = self.dataset[:min(limit, len(self.dataset))]
            logger.info(f"Limited to {len(data)} samples")
        
        for sample in data:
            yield sample
    
    def get_prompt(self, sample: Dict) -> Tuple[str, str]:
        """Build prompts for a sample."""
        system_prompt = self.config.get("prompts", {}).get("system", "")
        
        text = sample.get("text", "")
        choices = sample.get("choices", [])
        
        choice_letters = ['A', 'B', 'C', 'D', 'E']
        choices_text = "\n".join(
            f"{choice_letters[i]}) {choice}"
            for i, choice in enumerate(choices)
        )
        
        user_prompt = f"{text}\n\nOpciones:\n{choices_text}\n\nRespuesta:"
        
        return system_prompt, user_prompt
    
    def get_task_name(self) -> str:
        """Return task identifier."""
        return "openbookqa_es"
