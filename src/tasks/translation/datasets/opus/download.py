"""Download and preprocess OPUS-100 dataset."""

import json
import logging
from pathlib import Path
from typing import Dict, List

try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

logger = logging.getLogger(__name__)


def download_and_preprocess_opus(
    dataset_name: str,
    language_pairs: List[str],
    output_dir: Path,
    cache_dir: str = "./cache",
    split: str = "test",
) -> Dict[str, int]:
    """Download and preprocess OPUS-100 dataset.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "Helsinki-NLP/opus-100")
        language_pairs: List of language pairs to extract (e.g., ["en-es", "en-pt"])
        output_dir: Directory to save JSONL files
        cache_dir: Cache directory for datasets library
        split: Dataset split to use (train, validation, test)
        
    Returns:
        Dictionary with counts per language pair
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library not available. Install with: pip install datasets")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    counts = {}
    
    for pair in language_pairs:
        src_lang, tgt_lang = pair.split("-")
        logger.info(f"Downloading OPUS-100 {pair} ({split} split)...")
        
        try:
            # Load dataset for this language pair
            dataset = datasets.load_dataset(
                dataset_name,
                f"{src_lang}-{tgt_lang}",
                cache_dir=cache_dir,
                split=split,
            )
            
            # Convert to JSONL
            output_file = output_dir / f"{pair}.jsonl"
            sample_count = 0
            
            with open(output_file, "w", encoding="utf-8") as f:
                for idx, doc in enumerate(dataset):
                    # OPUS format: doc['translation'] is a dict with language codes as keys
                    translation = doc.get("translation", {})
                    src_text = translation.get(src_lang, "")
                    tgt_text = translation.get(tgt_lang, "")
                    
                    if src_text and tgt_text:
                        sample = {
                            "id": f"opus_{pair}_{idx}",
                            "source_text": src_text,
                            "target_text": tgt_text,
                            "source_lang": src_lang,
                            "target_lang": tgt_lang,
                            "language_pair": pair,
                        }
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        sample_count += 1
            
            counts[pair] = sample_count
            logger.info(f"Saved {sample_count} samples to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {pair}: {e}")
            counts[pair] = 0
    
    return counts





