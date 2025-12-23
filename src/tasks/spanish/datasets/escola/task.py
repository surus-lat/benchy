"""EsCoLA task implementation.

Spanish grammatical acceptability task.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Any, Tuple

from ...base import SpanishTaskBase
from .....common.dataset_utils import download_huggingface_dataset, save_to_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / ".data" / "escola"


class EscolaTask(SpanishTaskBase):
    """EsCoLA task: Spanish grammatical acceptability."""
    
    def __init__(self, config: Dict):
        """Initialize the EsCoLA task."""
        super().__init__(config)
        
        dataset_config = config.get("dataset", {})
        self.dataset_path = dataset_config.get("dataset_path", "nbel/EsCoLA")
        self.split = dataset_config.get("split", "validation")
        
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
                    self.dataset.append(json.loads(line))
        
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
            sentence = raw_sample.get("Sentence", "")
            label = raw_sample.get("Label", 0)
            
            # Build prompt text
            prompt_text = f"{sentence}\nPregunta: ¿Tiene sentido esta frase?\nRespuesta:"
            
            # Choices are ["no", "sí"]
            choices = ["no", "sí"]
            
            processed.append({
                "id": raw_sample.get("idx", f"escola_{idx}"),
                "text": prompt_text,
                "choices": choices,
                "expected": label,  # Label is already the index (0 or 1)
                "expected_idx": label,
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
        
        choice_letters = ['A', 'B', 'C', 'D']
        choices_text = "\n".join(
            f"{choice_letters[i]}) {choice}"
            for i, choice in enumerate(choices)
        )
        
        user_prompt = f"{text}\n\nOpciones:\n{choices_text}\n\nRespuesta:"
        
        return system_prompt, user_prompt
    
    def get_task_name(self) -> str:
        """Return task identifier."""
        return "escola"

