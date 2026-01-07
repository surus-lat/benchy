"""SimpleTask base class for low-boilerplate tasks.

This class implements the BaseTask protocol with sensible defaults:
- Loads JSONL or HuggingFace datasets.
- Builds prompts from a template.
- Computes metrics from a registry of Metric objects.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..common.dataset_utils import download_huggingface_dataset, iterate_samples, load_jsonl_dataset, save_to_jsonl
from ..engine.protocols import BaseTask
from .metrics import Metric

logger = logging.getLogger(__name__)


class SimpleTask(BaseTask):
    """Base class that implements BaseTask with minimal overrides.

    Typical usage is to set `name`, `metrics`, and override preprocess_sample().
    More advanced tasks can override load_dataset() or aggregate_metrics().
    """

    name: str = "simple_task"
    metrics: List[Metric] = []
    answer_type: str = "freeform"
    requires_logprobs: bool = False
    prefers_logprobs: bool = False
    requires_schema: bool = False
    requires_multimodal: bool = False
    dataset_split: str = "train"
    default_data_file: str = "data.jsonl"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset: Optional[List[Dict[str, Any]]] = None

        dataset_config = config.get("dataset", {})
        # Resolve local data paths for caching and JSONL loading.
        self.data_dir = self._resolve_data_dir(dataset_config.get("data_dir"))
        self.data_file = self._resolve_data_file(dataset_config.get("data_file"))

    def _resolve_data_dir(self, override: Optional[str]) -> Path:
        # Default to a `.data` directory adjacent to the task module.
        if override:
            return Path(override)
        try:
            task_file = Path(inspect.getfile(self.__class__)).resolve()
            return task_file.parent / ".data"
        except (TypeError, OSError):
            return Path(".data")

    def _resolve_data_file(self, data_file: Optional[str]) -> Path:
        # Resolve relative dataset paths against the data_dir.
        filename = data_file or self.default_data_file
        candidate = Path(filename)
        if candidate.is_absolute():
            return candidate
        return self.data_dir / candidate

    def load(self) -> None:
        """Load dataset, downloading if configured."""
        # Keep the dataset on self so get_samples() can iterate.
        self.dataset = self.load_dataset()

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from JSONL or HuggingFace.

        If a JSONL exists at data_file, it is loaded directly. Otherwise,
        dataset_path/dataset_name is used to download and preprocess samples.
        """
        dataset_config = self.config.get("dataset", {})
        dataset_path = dataset_config.get("dataset_path") or dataset_config.get("dataset_name")
        split = dataset_config.get("split", self.dataset_split)
        cache_dir = dataset_config.get("cache_dir", str(self.data_dir / "cache"))

        if self.data_file.exists():
            logger.info(f"Loading dataset from {self.data_file}")
            return load_jsonl_dataset(self.data_file)

        if not dataset_path:
            raise FileNotFoundError(
                f"Dataset file not found and no dataset_path configured: {self.data_file}"
            )

        logger.info(f"Downloading dataset {dataset_path} ({split})")
        raw_samples = download_huggingface_dataset(
            dataset_name=dataset_path,
            split=split,
            cache_dir=cache_dir,
        )

        processed: List[Dict[str, Any]] = []
        skipped = 0
        # Apply task-specific preprocessing to raw HF samples.
        for idx, raw_sample in enumerate(raw_samples):
            sample = self.preprocess_sample(raw_sample, idx)
            if sample is None:
                skipped += 1
                continue
            if "id" not in sample:
                sample["id"] = f"sample_{len(processed)}"
            processed.append(sample)

        logger.info(f"Processed {len(processed)} samples, skipped {skipped}")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        save_to_jsonl(processed, self.data_file)
        return processed

    def preprocess_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """Transform a raw dataset sample to eval format.

        Override this to normalize fields into the expected eval schema.
        """
        sample = dict(raw_sample)
        if "id" not in sample:
            sample["id"] = f"sample_{idx}"
        return sample

    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset samples."""
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return iterate_samples(self.dataset, limit=limit)

    def get_prompt(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Build prompts for LLM interfaces.

        Uses the prompt templates in task.json and formats them with sample fields.
        """
        prompts = self.config.get("prompts", {})
        system_prompt = prompts.get("system", "")
        user_template = prompts.get("user", "{text}")
        try:
            user_prompt = user_template.format(**sample)
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(f"Missing prompt field '{missing}' in sample {sample.get('id')}") from exc
        return system_prompt, user_prompt

    def get_task_name(self) -> str:
        return self.name

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict[str, Any],
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Convert errors into a uniform metrics shape.
        if error or prediction is None:
            return self.get_error_metrics(error or "No prediction", error_type)

        metric_values: Dict[str, Any] = {}
        valid = True
        # Merge per-sample outputs from each metric.
        for metric in self.metrics:
            result = metric.per_sample(prediction, expected, sample)
            if "valid" in result:
                valid = valid and bool(result["valid"])
            metric_values.update(result)

        return {
            "valid": valid,
            **metric_values,
        }

    def aggregate_metrics(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Provide a sane baseline for empty datasets.
        if not all_metrics:
            return {
                "total_samples": 0,
                "valid_samples": 0,
                "error_count": 0,
            }

        # Filter to valid samples and aggregate metric outputs.
        valid = [m for m in all_metrics if m.get("valid")]
        summary: Dict[str, Any] = {
            "total_samples": len(all_metrics),
            "valid_samples": len(valid),
            "error_count": len(all_metrics) - len(valid),
        }
        # Derived error_rate is useful across many tasks.
        summary["error_rate"] = (
            summary["error_count"] / summary["total_samples"] if summary["total_samples"] else 0.0
        )

        for metric in self.metrics:
            summary.update(metric.aggregate(valid))

        return summary

    def get_error_metrics(
        self,
        error: str,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        metric_defaults = {metric.name: 0.0 for metric in self.metrics}
        return {
            "valid": False,
            "error": error,
            "error_type": error_type,
            **metric_defaults,
        }
