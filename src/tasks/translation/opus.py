"""OPUS-100 translation subtask."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ._translation_handler import TranslationHandler
from .datasets.opus.download import download_and_preprocess_opus

logger = logging.getLogger(__name__)


class Opus(TranslationHandler):
    """OPUS-100 English-centric translation task.
    
    Evaluates English-Spanish and English-Portuguese translation pairs
    from the OPUS-100 dataset (Helsinki-NLP/opus-100).
    
    The dataset is preprocessed into bidirectional pairs:
    - en->es and es->en
    - en->pt and pt->en
    
    Scores are averaged across both directions per language pair.
    """
    
    # Task metadata
    name = "opus"
    display_name = "OPUS-100 Translation"
    description = "OPUS-100 English-centric translation evaluation"
    
    # Dataset configuration
    dataset_name = "Helsinki-NLP/opus-100"
    split = "test"
    language_pairs = ["en-es", "en-pt"]  # Default pairs
    
    # Prompts
    system_prompt = "You are a professional translator. Translate the given text accurately and naturally. Answer only with the translation, no other text or explanation is allowed."
    user_prompt_template = "Translate this sentence from {source_language} to {target_language}: {source_text}\n\nTranslation:"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Set up the OPUS-100 translation task and configure its data and cache directories.
        
        Parameters:
            config: Optional configuration dictionary. Recognized keys:
                - "comet_model": a preloaded COMET model instance used elsewhere by the task.
                - "language_pairs": list of language-pair strings to override the default pairs.
                - "dataset_name": HuggingFace dataset identifier to override the default dataset.
        """
        super().__init__(config)
        
        # Override language pairs if provided in config
        if self.config:
            self.language_pairs = self.config.get("language_pairs", self.language_pairs)
            self.dataset_name = self.config.get("dataset_name", self.dataset_name)
        
        # Data directory
        self.data_dir = Path(__file__).parent / ".data" / "opus"
        self.cache_dir = Path(__file__).parent / "cache"
    
    def load_dataset(self) -> List[Dict]:
        """
        Load and return OPUS-100 translation samples for all configured language pairs.
        
        This ensures the task data directory exists, triggers preprocessing/download for any missing language-pair files, and aggregates samples from each pair. Each returned sample is normalized and preprocessed to match the evaluation engine's expected fields.
        
        Returns:
            List[Dict]: A list of translation sample dictionaries. Each sample includes fields such as
                `id`, `text` (source text), `expected` (reference translation), `source_language`,
                `target_language`, `language_pair`, and `direction`.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure all language pairs are preprocessed
        for pair in self.language_pairs:
            pair_file = self.data_dir / f"{pair}.jsonl"
            
            if not pair_file.exists():
                logger.info(f"Preprocessing OPUS data for {pair}...")
                download_and_preprocess_opus(
                    dataset_name=self.dataset_name,
                    language_pairs=[pair],
                    output_dir=self.data_dir,
                    cache_dir=str(self.cache_dir),
                    split=self.split,
                )
        
        # Load all language pairs into one dataset
        # Note: We load ALL samples from ALL pairs, then the engine applies limit
        # This ensures balanced representation across language pairs
        all_samples = []
        samples_per_pair = {}
        
        for pair in self.language_pairs:
            pair_file = self.data_dir / f"{pair}.jsonl"
            
            if not pair_file.exists():
                logger.warning(f"Data file not found for {pair}: {pair_file}")
                continue
            
            pair_samples = []
            with open(pair_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    try:
                        raw_sample = json.loads(line)
                        # Ensure language_pair field is normalized
                        if "language_pair" not in raw_sample:
                            raw_sample["language_pair"] = pair.replace("-", "_")
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
        
        logger.info(f"Loaded {len(all_samples)} OPUS samples across {len(self.language_pairs)} language pairs")
        logger.info(f"Samples per pair: {samples_per_pair}")
        return all_samples
    
    def get_prompt(self, sample: Dict) -> Tuple[str, str]:
        """
        Constructs the system and user prompts for translating a single sample.
        
        Parameters:
            sample (dict): Sample containing `source_language`, `target_language`, and `source_text`. Missing fields default to `"Unknown"` for languages and `""` for text.
        
        Returns:
            tuple: (system_prompt, user_prompt) where `user_prompt` is formatted to request a translation from the sample's source language to its target language using the sample text.
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
        """
        Normalize a raw OPUS sample into the structure expected by the evaluation engine.
        
        Parameters:
            raw_sample (Dict[str, Any]): Original sample from the OPUS dataset; may contain fields like `id`, `source_text`, `target_text`, `source_language`, and `target_language`.
            idx (int): Index of the sample in the source file or batch; provided for context but not modified.
        
        Returns:
            Dict[str, Any]: The input sample copied and normalized so that `text` contains the source/input text (populated from `source_text` if missing) and `expected` contains the reference translation (populated from `target_text` if missing).
        """
        # OPUS samples already have id, source_text, target_text, etc.
        # Map to engine-expected fields
        sample = dict(raw_sample)
        
        # Map source_text -> text (input for generation)
        if "text" not in sample:
            sample["text"] = sample.get("source_text", "")
        
        # Map target_text -> expected (reference for metrics)
        if "expected" not in sample:
            sample["expected"] = sample.get("target_text", "")
        
        return sample
