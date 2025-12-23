"""PAWS-es task implementation.

Paraphrase identification in Spanish.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Any, Tuple

from ...base import SpanishTaskBase
from ...preprocessing import process_docs_paraphrases
from .....common.dataset_utils import download_huggingface_dataset, save_to_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / ".data" / "paws_es"


class PawsEsTask(SpanishTaskBase):
    """PAWS-es task: Paraphrase identification in Spanish."""
    
    def __init__(self, config: Dict):
        """Initialize the PAWS-es task."""
        super().__init__(config)
        
        dataset_config = config.get("dataset", {})
        self.dataset_path = dataset_config.get("dataset_path", "paws-x")
        self.dataset_name = dataset_config.get("dataset_name", "es")
        self.split = dataset_config.get("split", "test")
        
        self.data_file = DATA_DIR / f"{self.split}.jsonl"
        self.dataset: Optional[list] = None
    
    def load(self) -> None:
        """Load dataset, auto-downloading from HuggingFace if needed."""
        if not self.data_file.exists():
            logger.info(f"Data file not found: {self.data_file}")
            logger.info(f"Downloading from HuggingFace: {self.dataset_path} (config: {self.dataset_name})")
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
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        logger.info(f"Loading dataset {self.dataset_path} with config {self.dataset_name}")
        dataset = load_dataset(
            self.dataset_path, 
            self.dataset_name, 
            split=self.split, 
            cache_dir=str(DATA_DIR / "cache"),
            trust_remote_code=True
        )
        
        processed = []
        skipped = 0
        
        for idx, raw_sample in enumerate(dataset):
            # Apply preprocessing
            processed_doc = process_docs_paraphrases(raw_sample.copy())
            if processed_doc is None:
                skipped += 1
                continue
            
            sentence1 = processed_doc.get("sentence1", "")
            sentence2 = processed_doc.get("sentence2", "")
            label = processed_doc.get("label", 0)
            
            # Build choices (from doc_to_choice template)
            choice1 = f"{sentence1}, ¿verdad? No, {sentence2}"
            choice2 = f"{sentence1}, ¿verdad? Sí, {sentence2}"
            choices = [choice1, choice2]
            
            # doc_to_text is empty, so we use the choices directly
            # But we'll format it nicely for the prompt
            prompt_text = f"{sentence1}\n{sentence2}"
            
            processed.append({
                "id": processed_doc.get("id", f"paws_es_{idx}"),
                "text": prompt_text,
                "choices": choices,
                "expected": label,  # Label is the index (0 or 1)
                "expected_idx": label,
                "raw_sample": processed_doc,
            })
        
        logger.info(f"Processed {len(processed)} samples, skipped {skipped}")
        
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
        return "paws_es_spanish_bench"

