"""
Task Development Utilities
===========================

This module contains ALL shared utilities for task development.
**Check here before writing new code - it probably already exists!**

Quick Start
-----------
For a new multiple-choice task with cached JSONL dataset:

    from ..common import (
        MultipleChoiceHandler,
        CachedDatasetMixin,
        download_huggingface_dataset,
        save_to_jsonl,
    )
    
    class MyTask(CachedDatasetMixin, MultipleChoiceHandler):
        dataset_name = "org/dataset"
        dataset_file = "test.jsonl"
        labels = {0: "No", 1: "Yes"}
        
        def _download_and_cache(self, output_path):
            raw = download_huggingface_dataset(self.dataset_name, self.split)
            processed = [self._transform(s) for s in raw]
            save_to_jsonl(processed, output_path)
        
        def _transform(self, sample):
            return {
                "text": sample["question"],
                "choices": ["No", "Yes"],
                "expected": sample["label"],
            }

That's it! ~25 lines total.

Available Components
-------------------

Format Handlers (most common):
- BaseHandler: Abstract base for all handlers
- MultipleChoiceHandler: For MC tasks (most common!)
- StructuredHandler: For JSON extraction tasks
- FreeformHandler: For text generation tasks
- MultimodalStructuredHandler: For image→JSON tasks

Dataset Loaders (for caching):
- CachedDatasetMixin: Auto-cache JSONL datasets
- CachedTSVMixin: Auto-cache TSV datasets  
- CachedCSVMixin: Auto-cache CSV datasets

Metrics (for evaluation):
- ExactMatch: Exact string matching
- F1Score: Token-level F1 score
- MultipleChoiceAccuracy: MC accuracy with parsing

Utilities:
- download_huggingface_dataset: Download from HuggingFace Hub
- load_jsonl_dataset: Load JSONL from disk
- save_to_jsonl: Save list of dicts to JSONL
- parse_choice_prediction: Robust choice parsing
- format_choices: Format choices with labels
"""

# Format handlers (most frequently used)
from .base import BaseHandler
from .multiple_choice import MultipleChoiceHandler
from .structured import StructuredHandler
from .freeform import FreeformHandler
from .multimodal_structured import MultimodalStructuredHandler

# Dataset loaders (second most used)
from .dataset_loaders import (
    CachedDatasetMixin,
    CachedTSVMixin,
    CachedCSVMixin,
)

# Metrics (third most used)
from .metrics import (
    Metric,
    ScalarMetric,
    ExactMatch,
    F1Score,
    MultipleChoiceAccuracy,
    MeanSquaredError,
    PearsonCorrelation,
)

# Utilities (auxiliary but still important)
from .utils.dataset_utils import (
    download_huggingface_dataset,
    load_jsonl_dataset,
    save_to_jsonl,
    iterate_samples,
)

from .utils.choice_utils import (
    parse_choice_prediction,
    parse_choice_index,
    format_choices,
    normalize_text,
    extract_answer_segment,
)

# Structured extraction utilities (for advanced use)
from .utils.structured_metrics_calculator import MetricsCalculator
from .utils.partial_matching import PartialMatcher

# Text processing utilities
from .utils.text_utils import (
    normalize_spaces,
    remove_accents,
    extract_float_score,
    format_score_with_comma,
)

__all__ = [
    # Format handlers
    "BaseHandler",
    "MultipleChoiceHandler",
    "StructuredHandler",
    "FreeformHandler",
    "MultimodalStructuredHandler",
    # Dataset loaders
    "CachedDatasetMixin",
    "CachedTSVMixin",
    "CachedCSVMixin",
    # Metrics
    "Metric",
    "ScalarMetric",
    "ExactMatch",
    "F1Score",
    "MultipleChoiceAccuracy",
    "MeanSquaredError",
    "PearsonCorrelation",
    # Dataset utilities
    "download_huggingface_dataset",
    "load_jsonl_dataset",
    "save_to_jsonl",
    "iterate_samples",
    # Choice utilities
    "parse_choice_prediction",
    "parse_choice_index",
    "format_choices",
    "normalize_text",
    "extract_answer_segment",
    # Structured extraction utilities
    "MetricsCalculator",
    "PartialMatcher",
    # Text processing utilities
    "normalize_spaces",
    "remove_accents",
    "extract_float_score",
    "format_score_with_comma",
]
