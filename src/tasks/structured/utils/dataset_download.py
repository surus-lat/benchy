"""Generic dataset download and preprocessing utilities."""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, Any, Optional

from datasets import load_dataset
from jsonschema import validate, ValidationError

from ..utils.schema_utils import sanitize_schema_for_vllm
from ..metrics.schema_complexity import compute_schema_complexity, compute_complexity_score

logger = logging.getLogger(__name__)


def download_and_preprocess_dataset(
    dataset_name: str,
    output_file: Path,
    cache_dir: str = "./cache",
    split: str = "train",
    max_input_chars: int = 20000,
    process_sample_fn: Optional[Callable[[Dict[str, Any], int], Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Download and preprocess a HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        output_file: Path to save processed JSONL
        cache_dir: HuggingFace cache directory
        split: Dataset split to use
        max_input_chars: Maximum input length in characters
        process_sample_fn: Optional custom processing function
        
    Returns:
        Dictionary with processing statistics
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    logger.info(f"Downloaded {len(dataset)} samples")
    
    processed_count = 0
    skipped_count = 0
    skipped_reasons = {
        "non_dict_expected": 0,
        "schema_validation_failed": 0,
        "parse_error": 0,
        "too_long": 0,
    }
    
    with open(output_file, "w") as f:
        for idx, sample in enumerate(dataset):
            try:
                # Use custom processing if provided, otherwise use default
                if process_sample_fn:
                    processed = process_sample_fn(sample, idx)
                else:
                    processed = _default_process_sample(
                        sample, idx, max_input_chars, skipped_reasons
                    )
                
                if processed is None:
                    skipped_count += 1
                    continue
                
                # Add sequential ID
                processed["id"] = f"sample_{processed_count}"
                
                f.write(json.dumps(processed) + "\n")
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} samples (skipped {skipped_count})...")
                    
            except Exception as e:
                logger.warning(f"Failed to process sample {idx}: {e}")
                skipped_count += 1
                skipped_reasons["parse_error"] += 1
    
    logger.info(f"Successfully saved {processed_count} samples to {output_file}")
    logger.info(f"Skipped {skipped_count} samples:")
    for reason, count in skipped_reasons.items():
        if count > 0:
            logger.info(f"  - {reason}: {count}")
    
    return {
        "processed": processed_count,
        "skipped": skipped_count,
        "reasons": skipped_reasons,
    }


def _default_process_sample(
    sample: Dict,
    idx: int,
    max_input_chars: int,
    skipped_reasons: Dict,
) -> Dict[str, Any]:
    """Default processing for paraloq dataset format."""
    # Parse schema and item
    schema = json.loads(sample["schema"]) if isinstance(sample["schema"], str) else sample["schema"]
    item = json.loads(sample["item"]) if isinstance(sample["item"], str) else sample["item"]
    
    # Skip non-dict outputs
    if not isinstance(item, dict):
        logger.debug(f"Skipping sample {idx}: expected is {type(item).__name__}, not dict")
        skipped_reasons["non_dict_expected"] += 1
        return None
    
    # Sanitize schema
    sanitized_schema = sanitize_schema_for_vllm(schema)
    
    # Compute complexity
    complexity_features = compute_schema_complexity(sanitized_schema)
    complexity_score = compute_complexity_score(complexity_features)
    
    # Check length
    text_len = len(sample["text"])
    schema_len = len(json.dumps(sanitized_schema))
    if text_len + schema_len > max_input_chars:
        logger.debug(f"Skipping sample {idx}: too long ({text_len + schema_len} chars)")
        skipped_reasons["too_long"] += 1
        return None
    
    # Validate expected against schema
    try:
        validate(instance=item, schema=sanitized_schema)
    except ValidationError as e:
        logger.debug(f"Skipping sample {idx}: validation failed: {e.message}")
        skipped_reasons["schema_validation_failed"] += 1
        return None
    
    return {
        "text": sample["text"],
        "schema": sanitized_schema,
        "expected": item,
        "title": sample.get("title", ""),
        "topic": sample.get("topic", ""),
        "medium": sample.get("medium", ""),
        "complexity": complexity_features,
        "complexity_score": complexity_score,
    }

