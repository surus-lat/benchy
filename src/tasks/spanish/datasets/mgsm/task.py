"""MGSM Direct Spanish task implementation.

Math word problems in Spanish.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Any, Tuple

from ...base import SpanishTaskBase
from .....common.dataset_utils import download_huggingface_dataset, save_to_jsonl

logger = logging.getLogger(__name__)

# Data directory relative to this module
DATA_DIR = Path(__file__).parent.parent.parent / ".data" / "mgsm"


class MgsmTask(SpanishTaskBase):
    """MGSM Direct Spanish task: Math word problems."""
    
    @property
    def task_type(self) -> str:
        """This is a generate_until task."""
        return "generate_until"
    
    def __init__(self, config: Dict):
        """Initialize the MGSM task.
        
        Args:
            config: Configuration dictionary with:
                - dataset.dataset_path: HuggingFace dataset path (default: "juletxara/mgsm")
                - dataset.dataset_name: Dataset name/config (default: "es")
                - dataset.split: Dataset split to use (default: "test")
                - prompts.system: System prompt (optional)
                - prompts.user: User prompt template (optional, will use default)
        """
        super().__init__(config)
        
        dataset_config = config.get("dataset", {})
        self.dataset_path = dataset_config.get("dataset_path", "juletxara/mgsm")
        self.dataset_name = dataset_config.get("dataset_name", "es")
        self.split = dataset_config.get("split", "test")
        
        # Data file path
        self.data_file = DATA_DIR / f"{self.split}.jsonl"
        
        self.dataset: Optional[list] = None
    
    def load(self) -> None:
        """Load dataset, auto-downloading from HuggingFace if needed."""
        # Auto-download if data file doesn't exist
        if not self.data_file.exists():
            logger.info(f"Data file not found: {self.data_file}")
            logger.info(f"Downloading from HuggingFace: {self.dataset_path} (config: {self.dataset_name})")
            self._download_and_preprocess()
        
        # Load from JSONL
        logger.info(f"Loading dataset from {self.data_file}")
        self.dataset = []
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.dataset.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.dataset)} samples")
    
    def _download_and_preprocess(self) -> None:
        """Download from HuggingFace and preprocess samples.
        
        The MGSM dataset is stored as a TSV file: mgsm_es.tsv
        Format: question<TAB>answer_number
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub required. Install with: pip install huggingface_hub")
        
        logger.info(f"Downloading MGSM TSV file for {self.dataset_name}/{self.split}")
        
        # Download the TSV file directly
        tsv_filename = f"mgsm_{self.dataset_name}.tsv"
        tsv_path = hf_hub_download(
            repo_id=self.dataset_path,
            filename=tsv_filename,
            repo_type="dataset",
            cache_dir=str(DATA_DIR / "cache"),
        )
        
        logger.info(f"Loading TSV file from {tsv_path}")
        
        # Parse TSV file
        dataset = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split by tab: question<TAB>answer_number
                parts = line.split("\t")
                if len(parts) >= 2:
                    question = parts[0].strip()
                    answer_number = parts[1].strip()
                    dataset.append({
                        "question": question,
                        "answer_number": answer_number,
                    })
        
        logger.info(f"Loaded {len(dataset)} samples from TSV file")
        
        # Preprocess each sample
        processed = []
        
        for idx, raw_sample in enumerate(dataset):
            question = raw_sample.get("question", "")
            answer_number = raw_sample.get("answer_number", "")
            
            if not question or not answer_number:
                logger.warning(f"Skipping sample {idx}: missing required fields")
                continue
            
            # Build prompt text (from doc_to_text template)
            # Use "Pregunta: " + question + "\nRespuesta: "
            prompt_text = f"Pregunta: {question}\nRespuesta: "
            
            processed.append({
                "id": f"mgsm_{idx}",
                "text": prompt_text,
                "expected": str(answer_number),  # Expected is the number as string
                "raw_sample": raw_sample,
            })
        
        logger.info(f"Processed {len(processed)} samples")
        
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
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.config.get("prompts", {}).get("system", "")
        user_prompt = sample.get("text", "")
        
        return system_prompt, user_prompt
    
    def get_task_name(self) -> str:
        """Return task identifier."""
        return "mgsm_direct_es_spanish_bench"

