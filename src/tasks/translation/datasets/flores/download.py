"""Download and preprocess FLORES+ dataset.

This module downloads FLORES+ from HuggingFace and processes it into
bidirectional language pairs, saving directly to our .data/flores/ directory.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    import datasets
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

logger = logging.getLogger(__name__)


# Language configurations (from build_dataset.py)
LANGUAGES = {
    'spa_Latn': 'Spanish',
    'por_Latn': 'Portuguese', 
    'eng_Latn': 'English',
    'fra_Latn': 'French',
    'ita_Latn': 'Italian',
    'deu_Latn': 'German',
    'hin_Deva': 'Hindi',
    'cmn_Hans': 'Chinese',
    'arb_Arab': 'Arabic',
}


def _get_language_pairs() -> List[tuple]:
    """Generate LATAM-focused bidirectional language pairs.
    
    All languages <-> Spanish + All languages <-> Portuguese
    (excluding spa-spa and por-por)
    
    Returns:
        List of (lang1, lang2) tuples
    """
    pairs = []
    
    # All languages <-> Spanish (excluding Spanish itself)
    for lang_code in LANGUAGES.keys():
        if lang_code != 'spa_Latn':
            pairs.append(('spa_Latn', lang_code))
    
    # All languages <-> Portuguese (excluding Portuguese itself and Spanish - already covered above)
    for lang_code in LANGUAGES.keys():
        if lang_code != 'por_Latn' and lang_code != 'spa_Latn':
            pairs.append(('por_Latn', lang_code))
    
    return pairs


def _get_pair_name(lang1: str, lang2: str) -> str:
    """Generate consistent pair name (alphabetical order).
    
    Args:
        lang1: First language code (e.g., 'spa_Latn')
        lang2: Second language code (e.g., 'eng_Latn')
        
    Returns:
        Pair name like 'eng_spa' (alphabetical)
    """
    lang1_short = lang1.split('_')[0]
    lang2_short = lang2.split('_')[0]
    
    # Always put in alphabetical order for consistency
    if lang1_short < lang2_short:
        return f"{lang1_short}_{lang2_short}"
    else:
        return f"{lang2_short}_{lang1_short}"


def _load_flores_dataset(cache_dir: Optional[str] = None):
    """Load the full FLORES+ dataset from HuggingFace.
    
    Args:
        cache_dir: Optional cache directory for datasets library
        
    Returns:
        DatasetDict with 'dev' and 'devtest' splits
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library not available. Install with: pip install datasets")
    
    logger.info("Downloading FLORES+ dataset from HuggingFace...")
    logger.info("This may take several minutes on first run...")
    
    dataset = datasets.load_dataset(
        'openlanguagedata/flores_plus',
        cache_dir=cache_dir,
    )
    
    logger.info("FLORES+ dataset downloaded successfully")
    return dataset


def _build_bidirectional_pair_dataset(
    full_dataset: datasets.DatasetDict,
    lang1: str,
    lang2: str,
    split: str
) -> List[Dict]:
    """Build bidirectional dataset for a specific language pair.
    
    Args:
        full_dataset: The full FLORES+ dataset
        lang1: First language code (e.g., 'spa_Latn')
        lang2: Second language code (e.g., 'eng_Latn')
        split: Dataset split ('dev' or 'devtest')
    
    Returns:
        List of translation documents for both directions
    """
    iso1 = lang1.split('_')[0]
    iso2 = lang2.split('_')[0]
    
    # Filter dataset for each language
    lang1_split = full_dataset[split].filter(lambda x: x['iso_639_3'] == iso1)
    lang2_split = full_dataset[split].filter(lambda x: x['iso_639_3'] == iso2)
    
    # Create lookup dictionaries
    lang1_lookup = {doc['id']: doc for doc in lang1_split}
    lang2_lookup = {doc['id']: doc for doc in lang2_split}
    
    # Find common document IDs
    common_ids = set(lang1_lookup.keys()) & set(lang2_lookup.keys())
    
    # Build bidirectional translation pairs
    translation_docs = []
    doc_id = 0
    
    for flores_id in sorted(common_ids, key=int):
        lang1_doc = lang1_lookup[flores_id]
        lang2_doc = lang2_lookup[flores_id]
        
        # Validate that documents have the same FLORES ID
        if lang1_doc['id'] != lang2_doc['id']:
            logger.warning(f"ID mismatch for {flores_id}: {lang1_doc['id']} != {lang2_doc['id']}")
            continue
        
        # Direction 1: lang1 -> lang2
        translation_docs.append({
            "id": str(doc_id),
            "flores_id": flores_id,
            "source_language": LANGUAGES[lang1],
            "source_text": lang1_doc['text'],
            "target_language": LANGUAGES[lang2],
            "target_text": lang2_doc['text'],
            "language_pair": f"{lang1}-{lang2}",
            "direction": f"{lang1}->{lang2}",
            "source_code": lang1,
            "target_code": lang2,
        })
        doc_id += 1
        
        # Direction 2: lang2 -> lang1
        translation_docs.append({
            "id": str(doc_id),
            "flores_id": flores_id,
            "source_language": LANGUAGES[lang2],
            "source_text": lang2_doc['text'],
            "target_language": LANGUAGES[lang1],
            "target_text": lang1_doc['text'],
            "language_pair": f"{lang1}-{lang2}",  # Keep consistent pair naming
            "direction": f"{lang2}->{lang1}",
            "source_code": lang2,
            "target_code": lang1,
        })
        doc_id += 1
    
    return translation_docs


def _save_dataset(docs: List[Dict], output_path: Path):
    """Save dataset as JSONL.
    
    Args:
        docs: List of document dictionaries
        output_path: Path to output JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    logger.debug(f"Saved {len(docs)} documents to {output_path}")


def download_and_preprocess_flores(
    dataset_name: str,
    language_pairs: Optional[List[str]],
    output_dir: Path,
    cache_dir: str = "./cache",
    use_existing_script: bool = False,  # Changed default - use direct implementation
    build_script_path: Optional[Path] = None,
) -> Dict[str, int]:
    """Download and preprocess FLORES+ dataset.
    
    This function:
    1. Downloads FLORES+ from HuggingFace
    2. Processes it into bidirectional language pairs
    3. Saves directly to output_dir (.data/flores/)
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "openlanguagedata/flores_plus")
        language_pairs: List of language pairs to filter (None = all pairs)
                       Format: ["eng_spa", "por_spa"] (short codes)
        output_dir: Where to save the data (src/tasks/translation/.data/flores/)
        cache_dir: Cache directory for datasets library
        use_existing_script: If True, try to use external build_dataset.py (legacy)
        build_script_path: Explicit path to build_dataset.py (legacy, unused if use_existing_script=False)
        
    Returns:
        Dictionary with counts per language pair
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets library not available. Install with: pip install datasets")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download FLORES+ from HuggingFace
    logger.info("Downloading FLORES+ dataset from HuggingFace...")
    full_dataset = _load_flores_dataset(cache_dir=cache_dir)
    
    # Step 2: Get all language pairs to process
    all_pairs = _get_language_pairs()
    
    # Filter pairs if language_pairs specified
    if language_pairs:
        # Convert short codes (eng_spa) to full codes (eng_Latn, spa_Latn)
        # Build mapping from short to full codes
        short_to_full = {code.split('_')[0]: code for code in LANGUAGES.keys()}
        
        filtered_pairs = []
        for pair_short in language_pairs:
            # Parse pair like "eng_spa" -> ("eng_Latn", "spa_Latn")
            parts = pair_short.split('_')
            if len(parts) == 2:
                lang1_short, lang2_short = parts
                if lang1_short in short_to_full and lang2_short in short_to_full:
                    lang1_full = short_to_full[lang1_short]
                    lang2_full = short_to_full[lang2_short]
                    # Check if this pair exists (in either order)
                    for l1, l2 in all_pairs:
                        if (l1 == lang1_full and l2 == lang2_full) or (l1 == lang2_full and l2 == lang1_full):
                            filtered_pairs.append((l1, l2))
                            break
        all_pairs = filtered_pairs if filtered_pairs else all_pairs
    
    logger.info(f"Processing {len(all_pairs)} bidirectional language pairs...")
    
    # Step 3: Process each language pair
    counts = {}
    
    for lang1, lang2 in all_pairs:
        pair_name = _get_pair_name(lang1, lang2)
        logger.info(f"Processing {LANGUAGES[lang1]} ↔ {LANGUAGES[lang2]} ({pair_name})...")
        
        for split in ['dev', 'devtest']:
            # Build bidirectional dataset for this pair and split
            docs = _build_bidirectional_pair_dataset(full_dataset, lang1, lang2, split)
            
            # Save to output directory
            output_path = output_dir / pair_name / f"{split}.jsonl"
            _save_dataset(docs, output_path)
            
            # Count samples (only count once per pair, using dev split)
            if split == 'dev':
                counts[pair_name] = len(docs)
                logger.info(f"  {pair_name}/{split}: {len(docs)} translation pairs")
    
    logger.info(f"✅ Dataset building complete!")
    logger.info(f"📁 Files saved to: {output_dir}")
    logger.info(f"📊 Total language pairs: {len(counts)}")
    
    return counts
