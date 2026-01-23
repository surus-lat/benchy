"""Dataset utilities for downloading and preprocessing."""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Iterator

def _ensure_pyarrow_compat_for_datasets() -> None:
    """Make newer pyarrow versions compatible with older datasets releases.

    Some older `datasets` versions import `pyarrow.PyExtensionType`, which was removed
    in newer `pyarrow` releases. `datasets` only needs an ExtensionType base class for
    defining its custom Arrow extension types, so aliasing is sufficient.
    """
    try:
        import pyarrow as pa

        if not hasattr(pa, "PyExtensionType") and hasattr(pa, "ExtensionType"):
            setattr(pa, "PyExtensionType", pa.ExtensionType)
    except Exception:
        # If pyarrow isn't installed or changes again, let datasets raise a
        # clearer error when imported.
        return


_ensure_pyarrow_compat_for_datasets()

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_jsonl_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of samples as dictionaries
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
    """Download dataset from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to use
        cache_dir: HuggingFace cache directory
        
    Returns:
        List of samples as dictionaries
    """
    logger.info(f"Downloading {dataset_name} dataset (split: {split})...")
    try:
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        dataset_list = list(dataset)
        logger.info(f"Downloaded {len(dataset_list)} samples")
        return dataset_list
    except Exception as exc:
        # Some datasets on the Hub are misconfigured such that `datasets.load_dataset()`
        # fails while trying to build splits (often due to mismatched CSV headers across files).
        # For the common simple case where a repo contains `{split}.csv` or `{split}.tsv`,
        # fall back to downloading that single file and parsing it directly.
        message = str(exc)
        looks_like_schema_mismatch = (
            "All the data files must have the same columns" in message
            or "column names don't match" in message
        )
        if not looks_like_schema_mismatch:
            raise

        try:
            from huggingface_hub import hf_hub_download
        except Exception:
            # No hub client available; re-raise the original error.
            raise

        def _read_delimited(path: str, *, delimiter: str) -> List[Dict[str, Any]]:
            rows: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                for row in reader:
                    if not row:
                        continue
                    # Normalize keys only if the file actually contains whitespace-padded headers.
                    if any(isinstance(k, str) and k != k.strip() for k in row.keys()):
                        row = {k.strip(): v for k, v in row.items()}
                    rows.append(dict(row))
            return rows

        for filename, delimiter in ((f"{split}.csv", ","), (f"{split}.tsv", "\t")):
            try:
                local_path = hf_hub_download(
                    repo_id=dataset_name,
                    repo_type="dataset",
                    filename=filename,
                    cache_dir=cache_dir,
                )
            except Exception:
                continue

            logger.warning(
                f"Falling back to direct file parsing for {dataset_name} split '{split}' via {filename}"
            )
            rows = _read_delimited(local_path, delimiter=delimiter)
            logger.info(f"Downloaded {len(rows)} samples via fallback")
            return rows

        # Fallback didn't find a matching split file; surface the original error.
        raise


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
    """Iterate over dataset samples with optional limit.
    
    Args:
        dataset: List of samples
        limit: Maximum number of samples to return
        
    Yields:
        Sample dictionaries
    """
    dataset_to_use = dataset
    if limit is not None:
        dataset_to_use = dataset[:min(limit, len(dataset))]
        logger.info(f"Limited to {len(dataset_to_use)} samples")
    
    for sample in dataset_to_use:
        yield sample
