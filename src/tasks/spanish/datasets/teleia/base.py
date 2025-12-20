"""Base class for teleia tasks."""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Any, Tuple

from ...base import SpanishTaskBase
from .....common.dataset_utils import download_huggingface_dataset, save_to_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / ".data" / "teleia"


class TeleiaTaskBase(SpanishTaskBase):
    """Base class for teleia tasks."""
    
    def __init__(self, config: Dict, dataset_name: str, process_fn):
        """Initialize the teleia task.
        
        Args:
            config: Configuration dictionary
            dataset_name: Dataset name/config (pce, cervantes_ave, or siele)
            process_fn: Function to process documents
        """
        super().__init__(config)
        
        self.dataset_path = config.get("dataset", {}).get("dataset_path", "migonsa/teleia")
        self.dataset_name = dataset_name
        self.split = config.get("dataset", {}).get("split", "test")
        self.process_fn = process_fn
        
        self.data_file = DATA_DIR / dataset_name / f"{self.split}.jsonl"
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
        for idx, raw_sample in enumerate(dataset):
            # Apply preprocessing
            processed_doc = self.process_fn(raw_sample)
            
            query = processed_doc.get("query", "")
            choices = processed_doc.get("choices", [])
            target = processed_doc.get("target", 0)
            
            processed.append({
                "id": raw_sample.get("id", f"teleia_{self.dataset_name}_{idx}"),
                "text": query,
                "choices": choices,
                "expected": target,
                "expected_idx": target,
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

