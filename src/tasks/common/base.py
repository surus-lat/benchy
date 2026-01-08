"""Base format handler protocol and common utilities.

This module defines the standard task interface that all format handlers implement.
Format handlers encapsulate common patterns for loading datasets, preprocessing samples,
building prompts, and calculating metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple

from .utils.dataset_utils import (
    download_huggingface_dataset,
    iterate_samples,
    load_jsonl_dataset,
    save_to_jsonl,
)

logger = logging.getLogger(__name__)


class FormatHandler(Protocol):
    """Protocol defining the standard task interface.
    
    All format handlers must implement these methods to be compatible with
    the benchmark engine. This protocol matches the BaseTask protocol from
    the engine for compatibility.
    """

    def load(self) -> None:
        """Load the dataset. Called once before evaluation starts."""
        ...

    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict]:
        """Iterate over dataset samples.
        
        Args:
            limit: Maximum number of samples to return (None for all)
            
        Yields:
            Sample dictionaries with at minimum: id, text, expected
        """
        ...

    def get_prompt(self, sample: Dict) -> Tuple[str, str]:
        """Build prompt messages for a sample.
        
        Args:
            sample: Sample dictionary from get_samples()
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        ...

    def get_task_name(self) -> str:
        """Get the task identifier for logging and checkpointing."""
        ...

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for a single prediction.
        
        Args:
            prediction: Model output
            expected: Expected output
            sample: Full sample dict
            error: Error message if generation failed
            error_type: Type of error (connectivity_error, invalid_response)
            
        Returns:
            Metrics dictionary
        """
        ...

    @property
    def answer_type(self) -> str:
        """Expected answer type: 'freeform', 'structured', or 'multiple_choice'."""
        ...

    @property
    def requires_logprobs(self) -> bool:
        """Whether this task requires logprobs-based scoring."""
        ...


class BaseHandler:
    """Base implementation providing common functionality for format handlers.
    
    This class implements the FormatHandler protocol with sensible defaults.
    Subclasses should set class attributes for configuration and override methods
    as needed for custom behavior.
    
    Class Attributes (override these):
        name: Task name (defaults to class name in snake_case)
        dataset: HuggingFace dataset path or None for local JSONL
        split: Dataset split to use (default: "test")
        text_field: Field name for input text (default: "text")
        label_field: Field name for expected output (default: "expected")
        answer_type: Type of answer expected (default: "freeform")
        requires_logprobs: Whether logprobs are required (default: False)
        prefers_logprobs: Whether logprobs are preferred (default: False)
        requires_multimodal: Whether multimodal inputs are required (default: False)
        requires_schema: Whether JSON schema is required (default: False)
        requires_files: Whether file inputs are required (default: False)
        system_prompt: Default system prompt (default: "")
        user_prompt_template: Template for user prompts (default: "{text}")
        metrics: List of Metric objects (default: [])
        default_data_file: Filename for cached JSONL (default: "data.jsonl")
    """

    # Configuration defaults (override in subclasses)
    name: str = "base_task"
    dataset: Optional[str] = None
    split: str = "test"
    text_field: str = "text"
    label_field: str = "expected"
    answer_type: str = "freeform"
    requires_logprobs: bool = False
    prefers_logprobs: bool = False
    requires_multimodal: bool = False
    requires_schema: bool = False
    requires_files: bool = False
    system_prompt: str = ""
    user_prompt_template: str = "{text}"
    metrics: List[Any] = []
    default_data_file: str = "data.jsonl"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the handler.
        
        Args:
            config: Optional configuration dict (for compatibility with legacy code)
        """
        self.config = config or {}
        self.dataset_data: Optional[List[Dict[str, Any]]] = None

        # Resolve data directory (adjacent to task module)
        self.data_dir = self._resolve_data_dir()
        self.data_file = self.data_dir / self.default_data_file

    def _resolve_data_dir(self) -> Path:
        """Resolve the data directory for this task.
        
        Returns:
            Path to .data directory adjacent to the task module
        """
        # Use class module location to find task directory
        import inspect
        module_file = inspect.getfile(self.__class__)
        task_dir = Path(module_file).parent
        return task_dir / ".data"

    def load(self) -> None:
        """Load the dataset."""
        self.dataset_data = self.load_dataset()

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load and preprocess the dataset.
        
        If dataset path is set, downloads from HuggingFace and caches locally.
        Otherwise loads from local JSONL file.
        
        Returns:
            List of preprocessed samples
        """
        # Check for cached data first
        if self.data_file.exists():
            logger.info(f"Loading cached dataset from {self.data_file}")
            return load_jsonl_dataset(self.data_file)

        # Download from HuggingFace if dataset path is set
        if self.dataset:
            logger.info(f"Downloading dataset {self.dataset} (split: {self.split})")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            raw_dataset = download_huggingface_dataset(
                dataset_name=self.dataset,
                split=self.split,
                cache_dir=str(self.data_dir / "hf_cache"),
            )

            # Preprocess samples
            processed = []
            for idx, raw_sample in enumerate(raw_dataset):
                sample = self.preprocess_sample(raw_sample, idx)
                if sample is not None:
                    processed.append(sample)

            # Cache for future use
            save_to_jsonl(processed, self.data_file)
            logger.info(f"Cached {len(processed)} samples to {self.data_file}")
            return processed

        # Load from local JSONL
        if self.data_file.exists():
            return load_jsonl_dataset(self.data_file)

        raise FileNotFoundError(
            f"No dataset found. Set 'dataset' attribute or provide {self.data_file}"
        )

    def preprocess_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """Transform a raw dataset sample to eval format.
        
        Override this to normalize fields into the expected eval schema.
        
        Args:
            raw_sample: Raw sample from dataset
            idx: Sample index
            
        Returns:
            Processed sample dict or None to skip
        """
        sample = dict(raw_sample)
        if "id" not in sample:
            sample["id"] = f"{self.name}_{idx}"
        return sample

    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset samples.
        
        Args:
            limit: Maximum number of samples to return
            
        Yields:
            Sample dictionaries
        """
        if self.dataset_data is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return iterate_samples(self.dataset_data, limit=limit)

    def get_prompt(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Build prompts for LLM interfaces.
        
        Uses system_prompt and user_prompt_template class attributes,
        formatting the template with sample fields.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        try:
            user_prompt = self.user_prompt_template.format(**sample)
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(
                f"Missing prompt field '{missing}' in sample {sample.get('id')}"
            ) from exc
        return self.system_prompt, user_prompt

    def get_task_name(self) -> str:
        """Get the task identifier.
        
        Returns:
            Task name (class name in snake_case if not overridden)
        """
        if self.name != "base_task":
            return self.name
        # Convert class name to snake_case
        import re
        name = self.__class__.__name__
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
        return name.replace("_handler", "")

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict[str, Any],
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for a single prediction.
        
        Default implementation uses the metrics list. Override for custom logic.
        
        Args:
            prediction: Model output
            expected: Expected output
            sample: Full sample dict
            error: Error message if generation failed
            error_type: Type of error
            
        Returns:
            Metrics dictionary
        """
        if error:
            return self.get_error_metrics(error, error_type)

        results = {"valid": True}
        for metric in self.metrics:
            metric_result = metric.compute(prediction, expected, sample)
            if isinstance(metric_result, dict):
                results.update(metric_result)
            else:
                results[metric.name] = metric_result

        return results

    def get_error_metrics(
        self, error: str, error_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return error metrics structure.
        
        Override this to match your task's metrics structure.
        
        Args:
            error: Error message
            error_type: Type of error
            
        Returns:
            Metrics dict with error information
        """
        return {
            "valid": False,
            "error": error,
            "error_type": error_type,
        }

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics across all samples.
        
        Default implementation computes means for numeric fields.
        Override for custom aggregation logic.
        
        Args:
            all_metrics: List of per-sample metric dicts
            
        Returns:
            Aggregated metrics dict
        """
        if not all_metrics:
            return {}

        aggregated = {}
        valid_samples = sum(1 for m in all_metrics if m.get("valid", True))
        aggregated["total_samples"] = len(all_metrics)
        aggregated["valid_samples"] = valid_samples
        aggregated["error_count"] = len(all_metrics) - valid_samples

        # Compute means for numeric metrics
        numeric_keys = set()
        for metrics in all_metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key not in ["valid"]:
                    numeric_keys.add(key)

        for key in numeric_keys:
            values = [m.get(key, 0) for m in all_metrics if m.get("valid", True)]
            if values:
                aggregated[key] = sum(values) / len(values)

        return aggregated

