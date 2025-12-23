"""Task-specific dataset download and preprocessing utilities for structured extraction."""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, Any, Optional

from datasets import load_dataset
from jsonschema import validate, ValidationError
from huggingface_hub import hf_hub_download

from ....common.dataset_utils import download_huggingface_dataset, save_to_jsonl
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
    """Download and preprocess a HuggingFace dataset (Paraloq-specific).
    
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
    
    # Use common utility to download dataset
    dataset = download_huggingface_dataset(dataset_name, split=split, cache_dir=cache_dir)
    
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


def download_chat_extraction_schema(
    dataset_name: str,
    schema_file: Path,
    cache_dir: str = "./cache",
) -> Dict:
    """Download schema file from HuggingFace dataset repository.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        schema_file: Path to save the schema JSON file
        cache_dir: HuggingFace cache directory
        
    Returns:
        Dictionary with download status
    """
    schema_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Downloading schema file from {dataset_name}...")
        downloaded_path = hf_hub_download(
            repo_id=dataset_name,
            filename="schema_expected_lead_data.json",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        
        # Copy to desired location
        import shutil
        shutil.copy2(downloaded_path, schema_file)
        logger.info(f"Schema file saved to {schema_file}")
        
        return {"success": True, "path": str(schema_file)}
    except Exception as e:
        logger.error(f"Failed to download schema file: {e}")
        return {"success": False, "error": str(e)}


def process_chat_extraction_sample(
    sample: Dict,
    idx: int,
    schema: Dict,
    max_input_chars: int,
    skipped_reasons: Dict,
) -> Dict[str, Any]:
    """Process a sample from chat_structured_extraction dataset.
    
    Args:
        sample: Raw sample from dataset
        idx: Sample index
        schema: Loaded schema for validation
        max_input_chars: Maximum input length
        skipped_reasons: Dictionary to track skip reasons
        
    Returns:
        Processed sample or None if skipped
    """
    try:
        # Extract user messages and expected output
        user_messages = sample.get("input", {}).get("user_messages", [])
        expected_lead_data = sample.get("output", {}).get("expected_lead_data", {})
        
        # Skip if no expected data
        if not expected_lead_data:
            logger.debug(f"Skipping sample {idx}: no expected_lead_data")
            skipped_reasons["no_expected_data"] = skipped_reasons.get("no_expected_data", 0) + 1
            return None
        
        # Skip if expected is not a dict
        if not isinstance(expected_lead_data, dict):
            logger.debug(f"Skipping sample {idx}: expected_lead_data is not dict")
            skipped_reasons["non_dict_expected"] += 1
            return None
        
        # Concatenate user messages
        if isinstance(user_messages, list):
            text = "\n".join(str(msg) for msg in user_messages)
        else:
            text = str(user_messages)
        
        # Validate expected against original schema first (before sanitization)
        try:
            validate(instance=expected_lead_data, schema=schema)
        except ValidationError as e:
            logger.debug(f"Skipping sample {idx}: validation failed: {e.message}")
            skipped_reasons["schema_validation_failed"] += 1
            return None
        except (TypeError, ValueError) as e:
            # Handle validation errors like "unhashable type: 'list'"
            # This can happen if schema structure is incompatible with jsonschema
            logger.debug(f"Skipping sample {idx}: schema validation error ({type(e).__name__}): {e}")
            skipped_reasons["schema_validation_failed"] += 1
            return None
        except Exception as e:
            # Handle any other unexpected validation errors
            logger.warning(f"Skipping sample {idx}: unexpected validation error ({type(e).__name__}): {e}")
            skipped_reasons["schema_validation_failed"] += 1
            return None
        
        # Sanitize schema for vLLM (after validation)
        try:
            sanitized_schema = sanitize_schema_for_vllm(schema)
        except Exception as e:
            logger.warning(f"Skipping sample {idx}: schema sanitization error: {e}")
            skipped_reasons["parse_error"] += 1
            return None
        
        # Compute complexity
        complexity_features = compute_schema_complexity(sanitized_schema)
        complexity_score = compute_complexity_score(complexity_features)
        
        # Check length
        text_len = len(text)
        schema_len = len(json.dumps(sanitized_schema))
        if text_len + schema_len > max_input_chars:
            logger.debug(f"Skipping sample {idx}: too long ({text_len + schema_len} chars)")
            skipped_reasons["too_long"] += 1
            return None
        
        # Extract metadata if available
        meta = sample.get("meta", {})
        
        return {
            "id": sample.get("id", f"sample_{idx}"),
            "text": text,
            "schema": sanitized_schema,
            "expected": expected_lead_data,
            "title": meta.get("source", "chat_extract"),
            "topic": "chat_lead_extraction",
            "medium": "conversational",
            "complexity": complexity_features,
            "complexity_score": complexity_score,
        }
    except Exception as e:
        logger.warning(f"Error processing chat extraction sample {idx}: {e}")
        skipped_reasons["parse_error"] += 1
        return None


def download_and_preprocess_chat_extraction(
    dataset_name: str,
    output_file: Path,
    schema_file: Path,
    cache_dir: str = "./cache",
    split: str = "train",
    max_input_chars: int = 20000,
) -> Dict[str, int]:
    """Download and preprocess chat_structured_extraction dataset.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        output_file: Path to save processed JSONL
        schema_file: Path to save schema JSON file
        cache_dir: HuggingFace cache directory
        split: Dataset split to use
        max_input_chars: Maximum input length in characters
        
    Returns:
        Dictionary with processing statistics
    """
    # First download the schema file
    schema_download_result = download_chat_extraction_schema(
        dataset_name=dataset_name,
        schema_file=schema_file,
        cache_dir=cache_dir,
    )
    
    if not schema_download_result.get("success"):
        raise RuntimeError(f"Failed to download schema: {schema_download_result.get('error')}")
    
    # Load the schema
    schema_file = '/home/mauro/dev/benchy/src/tasks/structured/.data/schema_expected_lead_data_modified.json'
    with open(schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)
    
    # Download dataset.jsonl file directly
    logger.info(f"Downloading {dataset_name} dataset...")
    
    # Try to download dataset.jsonl directly first (as specified by user)
    try:
        dataset_file_path = hf_hub_download(
            repo_id=dataset_name,
            filename="dataset.jsonl",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        # Load JSONL manually
        dataset = []
        with open(dataset_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        logger.info(f"Loaded dataset.jsonl with {len(dataset)} samples")
    except Exception as e:
        logger.warning(f"Could not download dataset.jsonl directly: {e}")
        # Fallback: try loading as a dataset with splits
        try:
            dataset_obj = load_dataset(dataset_name, cache_dir=cache_dir)
            # If it's a DatasetDict, get the default split
            if hasattr(dataset_obj, 'keys'):
                # Use 'train' split or first available
                if 'train' in dataset_obj:
                    dataset = list(dataset_obj['train'])
                else:
                    dataset = list(dataset_obj[list(dataset_obj.keys())[0]])
            else:
                dataset = list(dataset_obj)
            logger.info(f"Loaded dataset with {len(dataset)} samples")
        except Exception as e2:
            raise RuntimeError(f"Failed to load dataset: {e2}")
    
    processed_count = 0
    skipped_count = 0
    skipped_reasons = {
        "non_dict_expected": 0,
        "schema_validation_failed": 0,
        "parse_error": 0,
        "too_long": 0,
        "no_expected_data": 0,
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(dataset):
            processed = process_chat_extraction_sample(
                sample, idx, schema, max_input_chars, skipped_reasons
            )
            
            if processed is None:
                skipped_count += 1
                continue
            
            # Keep original ID if present
            if "id" not in processed or not processed["id"]:
                processed["id"] = f"sample_{processed_count}"
            
            f.write(json.dumps(processed, ensure_ascii=False) + "\n")
            processed_count += 1
            
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} samples (skipped {skipped_count})...")
    
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

