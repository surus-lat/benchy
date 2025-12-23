# Common Utilities

This directory contains shared utilities for benchmark tasks.

## Purpose

Common utilities provide reusable building blocks for:
- Dataset downloading and preprocessing

These utilities are task-agnostic and can be used across all benchmark implementations.

## Available Utilities

### dataset_utils.py

Generic dataset operations for HuggingFace datasets and JSONL files.

**Functions:**

```python
def load_jsonl_dataset(file_path: Path) -> List[Dict[str, Any]]
```
Load dataset from JSONL file. Returns list of sample dictionaries.

```python
def download_huggingface_dataset(
    dataset_name: str,
    split: str = "train",
    cache_dir: str = "./cache"
) -> List[Dict[str, Any]]
```
Download dataset from HuggingFace Hub. Returns list of samples.

```python
def save_to_jsonl(
    samples: List[Dict[str, Any]],
    output_file: Path,
    ensure_ascii: bool = False
) -> None
```
Save samples to JSONL file.

```python
def iterate_samples(
    dataset: List[Dict[str, Any]],
    limit: int = None
) -> Iterator[Dict[str, Any]]
```
Iterate over dataset with optional limit.

**Usage Example:**

```python
from src.common.dataset_utils import (
    download_huggingface_dataset,
    save_to_jsonl
)

# Download dataset
samples = download_huggingface_dataset(
    "my-dataset/benchmark-data",
    split="train",
    cache_dir="./cache"
)

# Process samples (task-specific)
processed = [preprocess(s) for s in samples]

# Save to JSONL
save_to_jsonl(processed, Path("./data/processed.jsonl"))
```

## Best Practices

**Dataset Utilities:**
1. Use `download_huggingface_dataset()` for initial download
2. Implement task-specific preprocessing logic separately
3. Save preprocessed data to JSONL for faster loading
4. Use `iterate_samples()` with `limit` for testing
