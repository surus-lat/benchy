"""Utility functions for task development.

This module contains helper utilities that are used by format handlers
and tasks, but are not part of the main task development API.

For most task development, you should import from the parent `common` module:

    from ..common import download_huggingface_dataset, parse_choice_prediction

Rather than importing directly from this utils module.
"""

from .dataset_utils import (
    download_huggingface_dataset,
    load_jsonl_dataset,
    save_to_jsonl,
    iterate_samples,
)

from .choice_utils import (
    parse_choice_prediction,
    parse_choice_index,
    format_choices,
    normalize_text,
    extract_answer_segment,
)

__all__ = [
    "download_huggingface_dataset",
    "load_jsonl_dataset",
    "save_to_jsonl",
    "iterate_samples",
    "parse_choice_prediction",
    "parse_choice_index",
    "format_choices",
    "normalize_text",
    "extract_answer_segment",
]

