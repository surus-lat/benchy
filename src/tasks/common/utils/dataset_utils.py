"""Dataset utilities for downloading and preprocessing."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Iterator

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_jsonl_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load samples from a JSON Lines (JSONL) file into a list of dictionaries.
    
    Parameters:
        file_path (Path): Path to a JSONL file where each non-empty line is a JSON object.
    
    Returns:
        List[Dict[str, Any]]: Parsed sample dictionaries, one per non-empty line in the file.
    
    Raises:
        FileNotFoundError: If `file_path` does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    
    logger.info(f"Loaded {len(dataset)} samples from {file_path}")
    return dataset


def download_huggingface_dataset(
    dataset_name: str,
    split: str = "train",
    cache_dir: str = "./cache"
) -> List[Dict[str, Any]]:
    """
    Download a specific split of a HuggingFace dataset and return its samples.
    
    Parameters:
        dataset_name (str): HuggingFace dataset identifier (e.g., "glue", "squad").
        split (str): Dataset split to load (default "train").
        cache_dir (str): Directory to use for the HuggingFace dataset cache.
    
    Returns:
        List[Dict[str, Any]]: List of dataset samples, each represented as a dictionary.
    """
    logger.info(f"Downloading {dataset_name} dataset (split: {split})...")
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    logger.info(f"Downloaded {len(dataset)} samples")
    return list(dataset)


def save_to_jsonl(
    samples: List[Dict[str, Any]],
    output_file: Path,
    ensure_ascii: bool = False
) -> None:
    """Save samples to JSONL file.
    
    Args:
        samples: List of sample dictionaries
        output_file: Path to output JSONL file
        ensure_ascii: Whether to ensure ASCII encoding
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=ensure_ascii) + "\n")
    
    logger.info(f"Saved {len(samples)} samples to {output_file}")


def iterate_samples(
    dataset: List[Dict[str, Any]],
    limit: int = None
) -> Iterator[Dict[str, Any]]:
    """
    Yield sample dictionaries from a dataset with an optional cap.
    
    Parameters:
        dataset (List[Dict[str, Any]]): List of sample dictionaries to iterate.
        limit (int, optional): Maximum number of samples to yield; if `None`, iterate all samples.
    
    Returns:
        Iterator[Dict[str, Any]]: An iterator that yields sample dictionaries from the dataset.
    """
    dataset_to_use = dataset
    if limit is not None:
        dataset_to_use = dataset[:min(limit, len(dataset))]
        logger.info(f"Limited to {len(dataset_to_use)} samples")
    
    for sample in dataset_to_use:
        yield sample
