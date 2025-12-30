"""Classification task implementation."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ...engine.protocols import BaseTask
from ...common import download_huggingface_dataset, save_to_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / ".data"
LABEL_PATTERN = re.compile(r"-?\d+")


class ClassifyTask(BaseTask):
    """Minimal classification task for multiple subtasks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        dataset_config = config.get("dataset", {})
        self.subtask_name = config.get("subtask_name", "classify")
        self.dataset_path = dataset_config.get("dataset_path") or dataset_config.get("dataset_name")
        self.dataset_split = dataset_config.get("split", "train")
        self.text_field = dataset_config.get("text_field", "text")
        self.label_field = dataset_config.get("label_field", "label")

        label_values = dataset_config.get("label_values")
        if isinstance(label_values, list):
            self.label_values = [int(value) for value in label_values]
            self.label_values_set = set(self.label_values)
        else:
            self.label_values = None
            self.label_values_set = None

        data_file = dataset_config.get("data_file")
        if data_file:
            self.data_file = Path(data_file)
        else:
            self.data_file = DATA_DIR / f"{self.subtask_name}.jsonl"

        self.dataset: Optional[List[Dict[str, Any]]] = None

    def load(self) -> None:
        """Load dataset, auto-downloading from HuggingFace if needed."""
        if not self.data_file.exists():
            if not self.dataset_path:
                raise ValueError("dataset_path is required to download data")
            logger.info(f"Data file not found: {self.data_file}")
            logger.info(f"Downloading from HuggingFace: {self.dataset_path}")
            self._download_and_preprocess()

        logger.info(f"Loading dataset from {self.data_file}")
        self.dataset = []
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.dataset.append(json.loads(line))

        logger.info(f"Loaded {len(self.dataset)} samples")

    def _download_and_preprocess(self) -> None:
        """Download from HuggingFace and preprocess samples into JSONL."""
        raw_samples = download_huggingface_dataset(
            dataset_name=self.dataset_path,
            split=self.dataset_split,
            cache_dir=str(DATA_DIR / "cache"),
        )

        processed = []
        skipped = 0

        for idx, raw_sample in enumerate(raw_samples):
            result = self.preprocess_sample(raw_sample, idx)
            if result is not None:
                processed.append(result)
            else:
                skipped += 1

        logger.info(f"Processed {len(processed)} samples, skipped {skipped}")

        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        save_to_jsonl(processed, self.data_file)

    def preprocess_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """Transform a HuggingFace sample to eval format."""
        text = raw_sample.get(self.text_field)
        label_raw = raw_sample.get(self.label_field)
        if text is None or label_raw is None:
            return None

        try:
            label = int(label_raw)
        except (TypeError, ValueError):
            return None

        if self.label_values_set and label not in self.label_values_set:
            return None

        return {
            "id": f"{self.subtask_name}_{idx}",
            "text": str(text),
            "expected": label,
        }

    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset samples."""
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        data = self.dataset
        if limit is not None:
            data = self.dataset[:min(limit, len(self.dataset))]
            logger.info(f"Limited to {len(data)} samples")

        for sample in data:
            yield sample

    def get_prompt(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Build prompts for a sample."""
        system_prompt = self.config.get("prompts", {}).get(
            "system",
            "You are a helpful assistant.",
        )
        user_template = self.config.get("prompts", {}).get("user", "{text}")

        user_prompt = user_template.format(text=sample.get("text", ""))
        return system_prompt, user_prompt

    def get_task_name(self) -> str:
        """Return task identifier for logging and checkpointing."""
        return f"classify_{self.subtask_name}"

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict[str, Any],
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for a single prediction."""
        if error or prediction is None:
            return self.get_error_metrics(
                error=error or "No prediction",
                error_type=error_type,
            )

        parsed = self._parse_prediction(prediction)
        if parsed is None:
            return self.get_error_metrics(
                error="Could not parse label from response",
                error_type="invalid_response",
            )

        try:
            expected_value = int(expected)
        except (TypeError, ValueError):
            expected_value = expected

        is_correct = parsed == expected_value
        return {
            "valid": True,
            "accuracy": 1.0 if is_correct else 0.0,
            "correct": is_correct,
        }

    def get_error_metrics(
        self,
        error: str,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get error metrics structure for failed predictions."""
        return {
            "valid": False,
            "error": error,
            "error_type": error_type,
            "accuracy": 0.0,
            "correct": False,
        }

    def aggregate_metrics(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate per-sample metrics into summary statistics."""
        if not all_metrics:
            return {
                "total_samples": 0,
                "valid_samples": 0,
                "accuracy": 0.0,
                "error_count": 0,
                "error_rate": 0.0,
            }

        valid = [m for m in all_metrics if m.get("valid")]
        accuracy = sum(m.get("accuracy", 0.0) for m in valid) / len(valid) if valid else 0.0
        error_count = len(all_metrics) - len(valid)

        return {
            "total_samples": len(all_metrics),
            "valid_samples": len(valid),
            "accuracy": accuracy,
            "error_count": error_count,
            "error_rate": error_count / len(all_metrics) if all_metrics else 0.0,
        }

    def _parse_prediction(self, prediction: Any) -> Optional[int]:
        if isinstance(prediction, bool):
            candidate = int(prediction)
        elif isinstance(prediction, int):
            candidate = prediction
        elif isinstance(prediction, float) and prediction.is_integer():
            candidate = int(prediction)
        else:
            text = str(prediction).strip()
            lowered = text.lower()
            if self.label_values_set == {0, 1}:
                if lowered in ("yes", "true"):
                    candidate = 1
                elif lowered in ("no", "false"):
                    candidate = 0
                else:
                    candidate = None
            else:
                candidate = None

            if candidate is None:
                match = LABEL_PATTERN.search(text)
                if not match:
                    return None
                candidate = int(match.group(0))

        if self.label_values_set and candidate not in self.label_values_set:
            return None

        return candidate

    @property
    def requires_multimodal(self) -> bool:
        return False

    @property
    def requires_schema(self) -> bool:
        return False

    @property
    def answer_type(self) -> str:
        return "freeform"

    @property
    def requires_logprobs(self) -> bool:
        return False
