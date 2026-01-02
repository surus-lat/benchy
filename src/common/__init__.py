"""Common utilities for benchy tasks.

Note: Checkpoint utilities have moved to src/engine/checkpoint.py
Imports here are kept for backward compatibility.
"""

# Checkpoint utils - re-export from engine for backward compatibility
from ..engine.checkpoint import (
    get_checkpoint_path,
    get_config_hash,
    save_checkpoint,
    load_checkpoint,
)

# Dataset utils - remain in common
from .dataset_utils import (
    load_jsonl_dataset,
    download_huggingface_dataset,
    save_to_jsonl,
    iterate_samples,
)
from .choice_utils import (
    CHOICE_LABELS,
    format_choices,
    parse_choice_index,
)

__all__ = [
    # Checkpoint (from engine)
    "get_checkpoint_path",
    "get_config_hash",
    "save_checkpoint",
    "load_checkpoint",
    # Dataset
    "load_jsonl_dataset",
    "download_huggingface_dataset",
    "save_to_jsonl",
    "iterate_samples",
    # Choice helpers
    "CHOICE_LABELS",
    "format_choices",
    "parse_choice_index",
]
