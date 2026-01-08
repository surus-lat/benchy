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
        """
        Yield dataset samples in order, optionally limited to a maximum count.
        
        Parameters:
            limit (Optional[int]): Maximum number of samples to yield; None yields all samples.
        
        Yields:
            Dict: Sample dictionaries containing at minimum `id`, `text`, and `expected`.
        """
        ...

    def get_prompt(self, sample: Dict) -> Tuple[str, str]:
        """
        Constructs the system and user prompts for a given sample.
        
        Parameters:
            sample (Dict): Sample dictionary whose fields are used to format the user prompt template.
        
        Returns:
            Tuple[str, str]: A pair (system_prompt, user_prompt). `system_prompt` is the handler's system_prompt; `user_prompt` is the user_prompt_template formatted with values from `sample`.
        
        Raises:
            KeyError: If a field required by the user_prompt_template is missing from `sample`. The exception message identifies the missing field and the sample id when available.
        """
        ...

    def get_task_name(self) -> str:
        """
        Return the task identifier used for logging and checkpointing.
        
        If the instance `name` attribute is not "base_task", that value is returned.
        Otherwise the identifier is derived from the handler class name by converting it to snake_case and removing a trailing `_handler` if present.
        
        Returns:
            task_name (str): The resolved task identifier.
        """
        ...

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute evaluation metrics for a single sample prediction, or return standardized error metrics when an error is provided.
        
        Parameters:
            prediction: The model's predicted output for the sample.
            expected: The ground-truth output for the sample.
            sample: The full sample dictionary (may be used by metrics).
            error: Optional error message describing a generation or evaluation failure.
            error_type: Optional short error category (e.g., "connectivity_error", "invalid_response").
        
        Returns:
            dict: Per-sample metrics. If `error` is provided, the dict will indicate the sample is invalid and include the error and `error_type` (e.g., {"valid": False, "error": ..., "error_type": ...}); otherwise it will include `"valid": True` and one or more metric values (either as named keys or nested dicts) produced by the configured metrics.
        """
        ...

    @property
    def answer_type(self) -> str:
        """
        Indicates the expected answer format for the task.
        
        Returns:
            One of 'freeform', 'structured', or 'multiple_choice', indicating how model responses should be interpreted.
        """
        ...

    @property
    def requires_logprobs(self) -> bool:
        """
        Indicates whether the handler requires logprobs-based scoring.
        
        Returns:
            `true` if logprobs-based scoring is required by the task, `false` otherwise.
        """
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
        """
        Initialize the handler, store an optional configuration, and set up dataset storage paths.
        
        Parameters:
            config (Optional[Dict[str, Any]]): Optional configuration overrides (legacy compatibility). If omitted, an empty configuration is used.
        
        Notes:
            - Resolves the task's `.data` directory adjacent to the task module and sets `self.data_dir`.
            - Sets `self.data_file` to `<data_dir>/<default_data_file>`.
        """
        self.config = config or {}
        self.dataset_data: Optional[List[Dict[str, Any]]] = None

        # Resolve data directory (adjacent to task module)
        self.data_dir = self._resolve_data_dir()
        self.data_file = self.data_dir / self.default_data_file

    def _resolve_data_dir(self) -> Path:
        """
        Resolve the data directory located adjacent to the task's module file.
        
        Determines the path by locating the file that defines the handler's class and returning its parent directory joined with ".data".
        
        Returns:
            Path to the ".data" directory adjacent to the task module.
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
        """
        Load the dataset, preprocess samples, and return the processed list.
        
        Attempts to load a cached JSONL at self.data_file. If not found and the handler's
        `dataset` attribute is set, downloads the dataset from HuggingFace, applies
        preprocess_sample to each item, caches the processed samples to self.data_file,
        and returns them. If neither a remote dataset nor a cached/local JSONL is
        available, raises FileNotFoundError.
        
        Returns:
            List[Dict[str, Any]]: A list of preprocessed sample dictionaries.
        
        Raises:
            FileNotFoundError: If no dataset is available via `self.dataset` and no local
            cached JSONL exists at self.data_file.
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
        """
        Convert a raw dataset record into the standardized evaluation sample format.
        
        If the input already contains the core fields "id" and "text", it is returned unchanged.
        Otherwise a shallow copy is made and an "id" is added if missing using the pattern
        "{self.name}_{idx}". Subclasses may override to extract or transform fields differently.
        
        Parameters:
            raw_sample (Dict[str, Any]): Raw sample from the dataset.
            idx (int): Index of the sample in the source dataset, used to generate an id when needed.
        
        Returns:
            Optional[Dict[str, Any]]: Processed sample dictionary, or `None` to skip the sample.
        """
        # If sample is already preprocessed (has id and text at minimum), return as-is
        # This allows load_dataset() to fully preprocess samples
        if "id" in raw_sample and "text" in raw_sample:
            return raw_sample
        
        # Otherwise, ensure it has an id
        sample = dict(raw_sample)
        if "id" not in sample:
            sample["id"] = f"{self.name}_{idx}"
        return sample

    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Yield samples from the already-loaded dataset.
        
        Parameters:
            limit (Optional[int]): Maximum number of samples to yield; if None, yields all samples.
        
        Returns:
            Iterator[Dict[str, Any]]: An iterator over sample dictionaries.
        
        Raises:
            RuntimeError: If the dataset has not been loaded via `load()`.
        """
        if self.dataset_data is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return iterate_samples(self.dataset_data, limit=limit)

    def get_prompt(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """
        Builds the system and user prompts for a sample.
        
        Formats the handler's user_prompt_template using fields from `sample` and returns the handler's system_prompt unchanged.
        
        Parameters:
            sample (Dict[str, Any]): Mapping of fields used to format `user_prompt_template`.
        
        Returns:
            Tuple[str, str]: A pair (system_prompt, user_prompt).
        
        Raises:
            KeyError: If `user_prompt_template` references a field missing from `sample`; the error message includes the missing field name and the sample id when available.
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
        """
        Return the task identifier used for logging and checkpointing.
        
        Returns:
            task_name (str): The configured `name` if it is not "base_task"; otherwise the class name converted to snake_case with a trailing "_handler" removed.
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
        """
        Compute per-sample metrics for a prediction using this handler's configured metrics.
        
        Parameters:
            prediction: The model's output for the sample.
            expected: The expected/ground-truth value for the sample.
            sample: The full sample dictionary (may be used by metric implementations).
            error: Optional error message indicating generation or processing failure.
            error_type: Optional classification of the error.
        
        Returns:
            A dictionary mapping metric names to computed values. Always includes a "valid" boolean and, when `error` is provided, contains error information (see `get_error_metrics`).
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
        """
        Return a standardized metrics dictionary representing a processing or evaluation error.
        
        Parameters:
            error (str): Human-readable error message describing the failure.
            error_type (Optional[str]): Optional machine-friendly error category or code.
        
        Returns:
            Dict[str, Any]: Metrics dictionary with keys:
                - `valid`: `False`.
                - `error`: the provided error message.
                - `error_type`: the provided error type or `None`.
        """
        return {
            "valid": False,
            "error": error,
            "error_type": error_type,
        }

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate per-sample metrics into overall metrics.
        
        Computes:
        - total_samples: total number of samples
        - valid_samples: count of samples where `"valid"` is True (assumed True if missing)
        - error_count: total_samples - valid_samples
        - mean values for any numeric metric keys (ints or floats, excluding `"valid"`) calculated over only the valid samples.
        
        Parameters:
            all_metrics (List[Dict]): List of per-sample metric dictionaries.
        
        Returns:
            Dict[str, Any]: Aggregated metrics dictionary containing totals and per-metric means.
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
