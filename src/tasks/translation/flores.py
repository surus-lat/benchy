"""FLORES+ translation subtask."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ._translation_handler import TranslationHandler
from .datasets.flores.download import download_and_preprocess_flores

logger = logging.getLogger(__name__)


# Default FLORES language pairs (30 total)
# All languages <-> Spanish + All languages <-> Portuguese
DEFAULT_LANGUAGE_PAIRS = [
    # Spanish pairs (8 languages)
    "arb_spa", "cmn_spa", "deu_spa", "eng_spa", 
    "fra_spa", "hin_spa", "ita_spa", "por_spa",
    # Portuguese pairs (7 languages, excluding spa)
    "arb_por", "cmn_por", "deu_por", "eng_por",
    "fra_por", "hin_por", "ita_por",
]


class Flores(TranslationHandler):
    """FLORES+ multilingual translation task.
    
    Evaluates high-quality translations across 30 language pairs focused on
    Latin American and Iberian languages. Based on FLORES-200 dataset.
    
    Covers bidirectional translation between:
    - Spanish <-> English, French, Italian, German, Hindi, Chinese, Arabic, Portuguese
    - Portuguese <-> English, French, Italian, German, Hindi, Chinese, Arabic
    
    Each language pair includes both directions (A->B and B->A).
    Scores are averaged across both directions per language pair.
    """
    
    # Task metadata
    name = "flores"
    display_name = "FLORES+ Translation"
    description = "FLORES+ multilingual translation evaluation"
    
    # Dataset configuration
    dataset_name = "openlanguagedata/flores_plus"
    split = "devtest"
    language_pairs = DEFAULT_LANGUAGE_PAIRS
    
    # Prompts
    system_prompt = "You are a professional translator. Translate the given text accurately and naturally. Answer only with the translation, no other text or explanation is allowed."
    user_prompt_template = "Translate this sentence from {source_language} to {target_language}: {source_text}\n\nTranslation:"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize FLORES translation task.
        
        Args:
            config: Configuration dict with:
                - comet_model: Preloaded COMET model
                - language_pairs: List of pairs to evaluate (optional)
                - split: Dataset split (dev or devtest, optional)
        """
        super().__init__(config)
        
        # Override configuration if provided
        if self.config:
            self.language_pairs = self.config.get("language_pairs", self.language_pairs)
            self.split = self.config.get("split", self.split)
        
        # Data directory
        self.data_dir = Path(__file__).parent / ".data" / "flores"
        self.cache_dir = Path(__file__).parent / "cache"
    
    def load_dataset(self) -> List[Dict]:
        """Load FLORES translation dataset.
        
        This loads all language pairs specified in self.language_pairs.
        Each pair's data is preprocessed from FLORES-200 into bidirectional pairs.
        
        The preprocessing:
        1. Downloads full FLORES+ dataset
        2. Matches documents by ID across languages
        3. Creates bidirectional pairs (A->B and B->A)
        4. Saves as JSONL per language pair
        
        Returns:
            List of translation samples with:
            - id, source_text, target_text
            - source_language, target_language
            - language_pair, direction, flores_id
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if preprocessing is needed
        missing_pairs = []
        for pair in self.language_pairs:
            pair_dir = self.data_dir / pair
            pair_file = pair_dir / f"{self.split}.jsonl"
            
            if not pair_file.exists():
                missing_pairs.append(pair)
        
        # Run preprocessing if any pairs are missing
        if missing_pairs:
            logger.info(f"Preprocessing FLORES data for {len(missing_pairs)} language pairs...")
            download_and_preprocess_flores(
                dataset_name=self.dataset_name,
                language_pairs=self.language_pairs,  # Process all pairs at once
                output_dir=self.data_dir,
                cache_dir=str(self.cache_dir),
            )
        
        # Load all language pairs into one dataset
        # Note: We load ALL samples from ALL pairs, then the engine applies limit
        # This ensures balanced representation across language pairs
        all_samples = []
        samples_per_pair = {}
        
        for pair in self.language_pairs:
            pair_dir = self.data_dir / pair
            pair_file = pair_dir / f"{self.split}.jsonl"
            
            if not pair_file.exists():
                logger.warning(f"Data file not found for {pair}: {pair_file}")
                continue
            
            pair_samples = []
            with open(pair_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    try:
                        raw_sample = json.loads(line)
                        # Preprocess to add required fields (text, expected)
                        sample = self.preprocess_sample(raw_sample, line_num)
                        if sample:
                            pair_samples.append(sample)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {pair_file}: {e}")
                        continue
            
            samples_per_pair[pair] = len(pair_samples)
            all_samples.extend(pair_samples)
            logger.info(f"Loaded {len(pair_samples)} samples for {pair}")
        
        logger.info(f"Loaded {len(all_samples)} FLORES samples across {len(self.language_pairs)} language pairs")
        logger.info(f"Samples per pair: {samples_per_pair}")
        return all_samples
    
    def get_prompt(self, sample: Dict) -> Tuple[str, str]:
        """Build translation prompt for a sample.
        
        Args:
            sample: Sample dict with source_language, target_language, source_text
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        source_lang = sample.get("source_language", "Unknown")
        target_lang = sample.get("target_language", "Unknown")
        source_text = sample.get("source_text", "")
        
        user_prompt = self.user_prompt_template.format(
            source_language=source_lang,
            target_language=target_lang,
            source_text=source_text,
        )
        
        return self.system_prompt, user_prompt
    
    def preprocess_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """Preprocess a sample for evaluation.
        
        FLORES samples are already preprocessed by download.py, but we need to
        ensure they have the 'text' and 'expected' fields for the engine.
        
        Args:
            raw_sample: Raw sample from dataset
            idx: Sample index
            
        Returns:
            Processed sample with required fields
        """
        # FLORES samples already have id, source_text, target_text, etc.
        # Map to engine-expected fields
        sample = dict(raw_sample)
        
        # Map source_text -> text (input for generation)
        if "text" not in sample:
            sample["text"] = sample.get("source_text", "")
        
        # Map target_text -> expected (reference for metrics)
        if "expected" not in sample:
            sample["expected"] = sample.get("target_text", "")
        
        return sample

