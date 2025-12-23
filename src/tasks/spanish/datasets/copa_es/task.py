"""COPA-es task implementation.

Causal reasoning task in Spanish.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Any, Tuple

from ...base import SpanishTaskBase
from .....common.dataset_utils import download_huggingface_dataset, save_to_jsonl

logger = logging.getLogger(__name__)

# Data directory relative to this module
DATA_DIR = Path(__file__).parent.parent.parent / ".data" / "copa_es"


class CopaEsTask(SpanishTaskBase):
    """COPA-es task: Causal reasoning in Spanish."""
    
    def __init__(self, config: Dict):
        """Initialize the COPA-es task.
        
        Args:
            config: Configuration dictionary with:
                - dataset.dataset_path: HuggingFace dataset path (default: "BSC-LT/COPA-es")
                - dataset.split: Dataset split to use (default: "test")
                - prompts.system: System prompt (optional)
                - prompts.user: User prompt template (optional, will use default)
        """
        super().__init__(config)
        
        dataset_config = config.get("dataset", {})
        self.dataset_path = dataset_config.get("dataset_path", "BSC-LT/COPA-es")
        self.split = dataset_config.get("split", "test")
        
        # Data file path
        self.data_file = DATA_DIR / f"{self.split}.jsonl"
        
        self.dataset: Optional[list] = None
    
    def load(self) -> None:
        """Load dataset, auto-downloading from HuggingFace if needed."""
        # Auto-download if data file doesn't exist
        if not self.data_file.exists():
            logger.info(f"Data file not found: {self.data_file}")
            logger.info(f"Downloading from HuggingFace: {self.dataset_path}")
            self._download_and_preprocess()
        
        # Load from JSONL
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
        from ...preprocessing import process_docs_copa_es
        
        # Download raw dataset
        raw_samples = download_huggingface_dataset(
            dataset_name=self.dataset_path,
            split=self.split,
            cache_dir=str(DATA_DIR / "cache"),
        )
        
        # Preprocess each sample
        processed = []
        skipped = 0
        
        for idx, raw_sample in enumerate(raw_samples):
            # Apply preprocessing
            processed_doc = process_docs_copa_es(raw_sample.copy())
            
            # Build prompt text (from doc_to_text template)
            premise = processed_doc.get("premise", "").rstrip(".!?,").strip()
            question = processed_doc.get("question", "")
            question_word = "porque" if question == "cause" else "y por lo tanto"
            prompt_text = f"{premise} {question_word}"
            
            # Get choices
            choice1 = processed_doc.get("choice1", "")
            choice2 = processed_doc.get("choice2", "")
            choices = [choice1, choice2]
            
            # Get expected answer (from doc_to_target)
            label = processed_doc.get("label", 0)
            expected = label
            
            processed.append({
                "id": processed_doc.get("idx", f"copa_es_{idx}"),
                "text": prompt_text,
                "choices": choices,
                "expected": expected,
                "expected_idx": label,
                "raw_sample": processed_doc,
            })
        
        logger.info(f"Processed {len(processed)} samples, skipped {skipped}")
        
        # Save to JSONL
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
        """Build prompts for a sample.
        
        Formats the prompt with choices as options.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.config.get("prompts", {}).get("system", "")
        
        # Format user prompt with choices
        text = sample.get("text", "")
        choices = sample.get("choices", [])
        
        # Format choices as A) choice1, B) choice2, etc.
        choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        choices_text = "\n".join(
            f"{choice_letters[i]}) {choice}"
            for i, choice in enumerate(choices)
        )
        
        user_prompt = f"{text}\n\nOpciones:\n{choices_text}\n\nRespuesta:"
        
        return system_prompt, user_prompt
    
    def get_task_name(self) -> str:
        """Return task identifier."""
        return "copa_es"
