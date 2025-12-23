"""WNLI-es task implementation.

Winograd NLI in Spanish.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Any, Tuple

from ...base import SpanishTaskBase
from .....common.dataset_utils import save_to_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / ".data" / "wnli_es"


class WnliEsTask(SpanishTaskBase):
    """WNLI-es task: Winograd NLI in Spanish."""
    
    def __init__(self, config: Dict):
        """Initialize the WNLI-es task."""
        super().__init__(config)
        
        dataset_config = config.get("dataset", {})
        self.dataset_path = dataset_config.get("dataset_path", "PlanTL-GOB-ES/wnli-es")
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
        """Download from HuggingFace and preprocess samples.
        
        The dataset is stored as CSV files:
        - wnli-dev-es.csv (validation split)
        - wnli-train-es.csv (train split)
        - wnli-test-shuffled-es.csv (test split)
        
        Format: index,sentence1,sentence2,label
        """
        try:
            from huggingface_hub import hf_hub_download
            import csv
        except ImportError:
            raise ImportError("huggingface_hub required. Install with: pip install huggingface_hub")
        
        # Map split names to CSV filenames
        split_to_file = {
            "validation": "wnli-dev-es.csv",
            "dev": "wnli-dev-es.csv",
            "train": "wnli-train-es.csv",
            "test": "wnli-test-shuffled-es.csv",
        }
        
        csv_filename = split_to_file.get(self.split, f"wnli-{self.split}-es.csv")
        logger.info(f"Downloading CSV file: {csv_filename}")
        
        # Download the CSV file directly
        csv_path = hf_hub_download(
            repo_id=self.dataset_path,
            filename=csv_filename,
            repo_type="dataset",
            cache_dir=str(DATA_DIR / "cache"),
        )
        
        logger.info(f"Loading CSV file from {csv_path}")
        
        # Parse CSV file
        processed = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sentence1 = row.get("sentence1", "").strip()
                sentence2 = row.get("sentence2", "").strip()
                label = int(row.get("label", 0))
                index = row.get("index", "")
                
                if not sentence1 or not sentence2:
                    continue
                
                # Build prompt text (from doc_to_text template)
                prompt_text = f"{sentence1}\nPregunta: {sentence2} ¿Verdadero o Falso?\nRespuesta:"
                
                # Choices are ["Falso", "Verdadero"]
                # Label 0 = Falso (neutral), Label 1 = Verdadero (entailment)
                choices = ["Falso", "Verdadero"]
                
                processed.append({
                    "id": index if index else f"wnli_es_{len(processed)}",
                    "text": prompt_text,
                    "choices": choices,
                    "expected": label,  # Label is the index (0 or 1)
                    "expected_idx": label,
                    "raw_sample": row,
                })
        
        logger.info(f"Processed {len(processed)} samples from CSV")
        
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
        
        WNLI may benefit from a system prompt explaining the task.
        Provide options in the prompt for letter-based multiple-choice scoring.
        """
        # Use system prompt from config, or default explanation
        system_prompt = self.config.get("prompts", {}).get(
            "system",
            "Eres un asistente experto en comprensión del lenguaje natural. "
            "Tu tarea es determinar si una segunda oración es una interpretación correcta de la primera."
        )
        
        text = sample.get("text", "")
        choices = sample.get("choices", [])
        
        choice_letters = ['A', 'B', 'C', 'D', 'E']
        choices_text = "\n".join(
            f"{choice_letters[i]}) {choice}"
            for i, choice in enumerate(choices)
        )
        
        # Strip any existing "Respuesta:" from the stored prompt text.
        if "Respuesta:" in text:
            text = text.rsplit("Respuesta:", 1)[0].rstrip()
        
        user_prompt = f"{text}\n\nOpciones:\n{choices_text}\n\nRespuesta:"
        
        return system_prompt, user_prompt
    
    def get_task_name(self) -> str:
        """Return task identifier."""
        return "wnli_es"
