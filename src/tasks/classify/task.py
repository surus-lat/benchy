"""Classification task implementation."""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ...engine.protocols import BaseTask
from ...common import CHOICE_LABELS, download_huggingface_dataset, format_choices, parse_choice_index, save_to_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / ".data"
LABEL_PATTERN = re.compile(r"-?\d+")
ANSWER_MARKERS = ("answer", "respuesta", "label", "etiqueta", "salida", "output")


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

        self.label_value_type = "numeric"
        self.label_values: Optional[List[Any]] = None
        self.label_values_set: Optional[set] = None
        self.label_text_to_value: Dict[str, Any] = {}
        self.label_texts: Dict[Any, str] = {}
        self.label_to_index: Dict[Any, int] = {}
        self.choice_texts: List[str] = []
        self.choice_labels: List[str] = []

        label_values = dataset_config.get("label_values")
        self._configure_label_values(label_values)
        self._load_label_texts(dataset_config.get("label_texts") or dataset_config.get("label_map"))
        self._configure_choices()

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

        self._load_from_file()
        if self._needs_rebuild(self.dataset):
            logger.info(f"Cached dataset is missing required fields: {self.data_file}")
            logger.info(f"Rebuilding dataset from HuggingFace: {self.dataset_path}")
            self._download_and_preprocess()
            self._load_from_file()

        logger.info(f"Loaded {len(self.dataset)} samples")

    def _load_from_file(self) -> None:
        logger.info(f"Loading dataset from {self.data_file}")
        self.dataset = []
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.dataset.append(json.loads(line))

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

        label = self._coerce_label_value(label_raw)
        if label is None:
            return None

        if self.label_values_set and label not in self.label_values_set:
            return None

        expected_idx = self.label_to_index.get(label)
        if expected_idx is None:
            return None

        return {
            "id": f"{self.subtask_name}_{idx}",
            "text": str(text),
            "expected": expected_idx,
            "choices": list(self.choice_texts),
            "choice_labels": list(self.choice_labels),
            "label_value": label,
        }

    def _needs_rebuild(self, dataset: Optional[List[Dict[str, Any]]]) -> bool:
        if not dataset:
            return True
        expected_labels = None
        if self.label_values:
            expected_labels = [str(value) for value in self.label_values]
        for sample in dataset:
            choices = sample.get("choices")
            expected = sample.get("expected")
            choice_labels = sample.get("choice_labels")
            if not isinstance(choices, list) or not choices:
                return True
            if not isinstance(expected, int):
                return True
            if expected < 0 or expected >= len(choices):
                return True
            if expected_labels:
                if not isinstance(choice_labels, list) or choice_labels != expected_labels:
                    return True
        return False

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

    def get_prompt_for_logprobs(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Build prompt optimized for logprobs scoring."""
        system_prompt = self.config.get("prompts", {}).get(
            "system",
            "You are a helpful assistant.",
        )
        user_template = self.config.get("prompts", {}).get("user", "{text}")
        base_prompt = user_template.format(text=sample.get("text", ""))
        choices_text = format_choices(sample.get("choices", []), sample.get("choice_labels"))
        user_prompt = f"{base_prompt}\n\nOptions:\n{choices_text}\n\nAnswer (label only):"
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

        parsed = self._parse_prediction(prediction, sample)
        if parsed is None:
            return self.get_error_metrics(
                error="Could not parse label from response",
                error_type="invalid_response",
            )

        is_correct = parsed == expected
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

    def _configure_label_values(self, label_values: Any) -> None:
        if not isinstance(label_values, list):
            return

        numeric_values = []
        all_numeric = True
        for value in label_values:
            try:
                numeric_values.append(int(value))
            except (TypeError, ValueError):
                all_numeric = False
                break

        if all_numeric:
            self.label_value_type = "numeric"
            self.label_values = numeric_values
            self.label_values_set = set(numeric_values)
            return

        text_values = [str(value) for value in label_values]
        self.label_value_type = "text"
        self.label_values = text_values
        self.label_values_set = set(text_values)
        for text in text_values:
            self._register_label_text(text, text)

    def _load_label_texts(self, label_texts: Any) -> None:
        if not label_texts:
            return

        if isinstance(label_texts, list):
            for idx, text in enumerate(label_texts):
                if text is None:
                    continue
                if self.label_values and idx < len(self.label_values):
                    label_value = self.label_values[idx]
                else:
                    label_value = idx
                self._register_label_text(label_value, text)
            return

        if isinstance(label_texts, dict):
            for key, value in label_texts.items():
                if isinstance(key, (int, float)) or (isinstance(key, str) and key.strip().lstrip("-").isdigit()):
                    try:
                        label_value = int(key)
                    except (TypeError, ValueError):
                        label_value = key
                    text = value
                else:
                    text = key
                    label_value = self._coerce_label_value(value)
                self._register_label_text(label_value, text)

    def _register_label_text(self, label_value: Any, text: Any) -> None:
        if label_value is None or text is None:
            return
        normalized = self._normalize_label_text(str(text))
        if not normalized:
            return
        self.label_text_to_value[normalized] = label_value
        self.label_texts[label_value] = str(text)

    def _normalize_label_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text)
        stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        lowered = stripped.lower()
        return re.sub(r"[^a-z0-9]+", " ", lowered).strip()

    def _coerce_label_value(self, value: Any) -> Optional[Any]:
        if value is None:
            return None
        if self.label_value_type == "numeric":
            try:
                return int(value)
            except (TypeError, ValueError):
                if isinstance(value, str):
                    normalized = self._normalize_label_text(value)
                    return self.label_text_to_value.get(normalized)
                return None

        text = str(value)
        normalized = self._normalize_label_text(text)
        if normalized in self.label_text_to_value:
            return self.label_text_to_value[normalized]
        return text

    def _extract_prediction_value(self, payload: Dict[str, Any]) -> Optional[Any]:
        for key in ("label", "answer", "prediction", "category", "class"):
            if key in payload:
                return payload[key]
        if len(payload) == 1:
            return next(iter(payload.values()))
        return None

    def _extract_answer_segment(self, text: str) -> str:
        lowered = text.lower()
        last_pos = -1
        for marker in ANSWER_MARKERS:
            idx = lowered.rfind(marker)
            if idx > last_pos:
                last_pos = idx
        if last_pos == -1:
            return text
        segment = text[last_pos:]
        split_idx = segment.find(":")
        if split_idx != -1:
            segment = segment[split_idx + 1 :]
        return segment.strip()

    def _parse_prediction(self, prediction: Any, sample: Dict[str, Any]) -> Optional[int]:
        if prediction is None:
            return None

        if isinstance(prediction, dict):
            extracted = self._extract_prediction_value(prediction)
            return self._parse_prediction(extracted, sample) if extracted is not None else None

        if isinstance(prediction, list):
            if len(prediction) == 1:
                return self._parse_prediction(prediction[0], sample)
            return None

        if isinstance(prediction, (bool, int, float)):
            choices = sample.get("choices", [])
            labels = sample.get("choice_labels")
            return parse_choice_index(
                prediction,
                choices,
                labels=labels,
                label_to_index=self.label_to_index,
            )

        else:
            text = str(prediction).strip()
            if not text:
                return None

            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = None
                if parsed is not None:
                    return self._parse_prediction(parsed, sample)

            answer_text = self._extract_answer_segment(text)
            prediction = answer_text

        choices = sample.get("choices", [])
        labels = sample.get("choice_labels")
        return parse_choice_index(
            prediction,
            choices,
            labels=labels,
            label_to_index=self.label_to_index,
        )

    @property
    def requires_multimodal(self) -> bool:
        return False

    @property
    def requires_schema(self) -> bool:
        return False

    @property
    def answer_type(self) -> str:
        return "multiple_choice"

    @property
    def requires_logprobs(self) -> bool:
        return False

    @property
    def prefers_logprobs(self) -> bool:
        return bool(self.choice_texts)

    def _configure_choices(self) -> None:
        if self.label_values:
            self.label_to_index = {value: idx for idx, value in enumerate(self.label_values)}
            for value in self.label_values:
                text = self.label_texts.get(value)
                if text is None:
                    text = str(value)
                self.choice_texts.append(str(text))
        elif self.label_texts:
            for key in sorted(self.label_texts.keys(), key=str):
                self.choice_texts.append(str(self.label_texts[key]))

        if self.label_values:
            self.choice_labels = [str(value) for value in self.label_values]
        else:
            self.choice_labels = CHOICE_LABELS[: len(self.choice_texts)]
