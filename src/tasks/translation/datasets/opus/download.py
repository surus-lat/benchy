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


# Language code to name mapping
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ar": "Arabic",
    "zh": "Chinese",
    "hi": "Hindi",
}


def download_and_preprocess_opus(
    dataset_name: str,
    language_pairs: List[str],
    output_dir: Path,
    cache_dir: str = "./cache",
    split: str = "test",
) -> Dict[str, int]:
    """
    Download and preprocess specified OPUS-100 language pairs into per-pair JSONL files.
    
    Each input example that contains both source and target text produces two samples (A->B and B->A) with metadata fields such as id, source_text, target_text, source_lang, target_lang, source_language, target_language, language_pair, and direction.
    
    Parameters:
        dataset_name (str): HuggingFace dataset identifier (e.g., "Helsinki-NLP/opus-100").
        language_pairs (List[str]): Language pair strings in the form "src-tgt" (e.g., ["en-es", "en-pt"]).
        output_dir (Path): Directory where per-pair JSONL files will be written; created if missing.
        cache_dir (str): Cache directory passed to the datasets library (default "./cache").
        split (str): Dataset split to use (e.g., "train", "validation", "test"; default "test").
    
    Returns:
        Dict[str, int]: Mapping from each language pair (as provided) to the number of generated samples.
    
    Raises:
        ImportError: If the HuggingFace datasets library is not available.
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
                        # Create bidirectional samples (A->B and B->A)
                        # Sample 1: src -> tgt
                        sample_ab = {
                            "id": f"opus_{pair}_{idx}_ab",
                            "source_text": src_text,
                            "target_text": tgt_text,
                            "source_lang": src_lang,
                            "target_lang": tgt_lang,
                            "source_language": LANGUAGE_NAMES.get(src_lang, src_lang),
                            "target_language": LANGUAGE_NAMES.get(tgt_lang, tgt_lang),
                            "language_pair": pair.replace("-", "_"),
                            "direction": f"{src_lang}->{tgt_lang}",
                        }
                        f.write(json.dumps(sample_ab, ensure_ascii=False) + "\n")
                        sample_count += 1
                        
                        # Sample 2: tgt -> src (reverse direction)
                        sample_ba = {
                            "id": f"opus_{pair}_{idx}_ba",
                            "source_text": tgt_text,
                            "target_text": src_text,
                            "source_lang": tgt_lang,
                            "target_lang": src_lang,
                            "source_language": LANGUAGE_NAMES.get(tgt_lang, tgt_lang),
                            "target_language": LANGUAGE_NAMES.get(src_lang, src_lang),
                            "language_pair": pair.replace("-", "_"),
                            "direction": f"{tgt_lang}->{src_lang}",
                        }
                        f.write(json.dumps(sample_ba, ensure_ascii=False) + "\n")
                        sample_count += 1
            
            counts[pair] = sample_count
            logger.info(f"Saved {sample_count} samples to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {pair}: {e}")
            counts[pair] = 0
    
    return counts




