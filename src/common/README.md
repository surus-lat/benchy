# Common Utilities

This directory contains shared utilities for benchmark tasks.

## Purpose

Common utilities provide reusable building blocks for:
- Dataset downloading and preprocessing
- Checkpoint management for resumable benchmarks
- File I/O operations

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

### checkpoint_utils.py

Checkpoint management for resumable benchmarks.

**Functions:**

```python
def get_checkpoint_path(
    output_dir: str,
    model_name: str,
    task_name: str
) -> Path
```
Get standardized checkpoint file path.

```python
def get_config_hash(config_dict: Dict[str, Any]) -> str
```
Generate MD5 hash of configuration for validation.

```python
def save_checkpoint(
    path: Path,
    completed_ids: List[str],
    config_hash: str
) -> None
```
Save checkpoint with completed sample IDs.

```python
def load_checkpoint(
    path: Path,
    expected_config_hash: str
) -> Set[str]
```
Load and validate checkpoint. Returns set of completed sample IDs.

**Usage Example:**

```python
from src.common.checkpoint_utils import (
    get_checkpoint_path,
    get_config_hash,
    save_checkpoint,
    load_checkpoint
)

# Setup checkpoint
checkpoint_path = get_checkpoint_path(
    output_dir="./results",
    model_name="my-model",
    task_name="my_task"
)

config_hash = get_config_hash({
    "model": "my-model",
    "temperature": 0.0,
    "batch_size": 20,
})

# Load existing checkpoint
completed_ids = load_checkpoint(checkpoint_path, config_hash)

# Process samples (skip completed)
for sample in samples:
    if sample["id"] not in completed_ids:
        # Process sample
        completed_ids.add(sample["id"])
        
        # Periodic checkpoint
        if len(completed_ids) % 50 == 0:
            save_checkpoint(
                checkpoint_path,
                list(completed_ids),
                config_hash
            )
```

## Best Practices

**Dataset Utilities:**
1. Use `download_huggingface_dataset()` for initial download
2. Implement task-specific preprocessing logic separately
3. Save preprocessed data to JSONL for faster loading
4. Use `iterate_samples()` with `limit` for testing

**Checkpoint Utilities:**
1. Include all relevant config in hash (model, temperature, etc.)
2. Checkpoint every 50-100 samples for long benchmarks
3. Delete checkpoint file on successful completion
4. Handle config changes gracefully (checkpoint invalidation)

**Task-Specific vs Common:**
- **Common**: Generic file I/O, dataset downloading, checkpointing
- **Task-Specific**: Dataset schema validation, metric calculation, prompt formatting

## Contributing

When adding new utilities:
1. Keep functions simple and focused
2. Add type hints to all parameters and return values
3. Document with clear docstrings
4. Avoid task-specific logic
5. Test with multiple tasks before committing

