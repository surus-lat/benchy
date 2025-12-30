"""Classification task implementation."""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ...engine.protocols import BaseTask
from ...common import download_huggingface_dataset, save_to_jsonl

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

        label_values = dataset_config.get("label_values")
        self._configure_label_values(label_values)
        self._load_label_texts(dataset_config.get("label_texts") or dataset_config.get("label_map"))

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

        label = self._coerce_label_value(label_raw)
        if label is None:
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

        expected_value = self._coerce_label_value(expected)
        if expected_value is None:
            return self.get_error_metrics(
                error="Invalid expected label",
                error_type="invalid_response",
            )

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

    def _match_label_text(self, text: str) -> Optional[Any]:
        if not self.label_text_to_value:
            return None
        normalized = self._normalize_label_text(text)
        if not normalized:
            return None
        if normalized in self.label_text_to_value:
            return self.label_text_to_value[normalized]
        best_value = None
        best_len = 0
        response_words = normalized.split()
        for label_text, value in self.label_text_to_value.items():
            if not label_text or label_text not in normalized:
                continue
            label_words = label_text.split()
            if response_words and len(response_words) > len(label_words) + 3:
                continue
            if len(label_text) > best_len:
                best_value = value
                best_len = len(label_text)
        return best_value

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

    def _parse_prediction(self, prediction: Any) -> Optional[Any]:
        if prediction is None:
            return None

        if isinstance(prediction, dict):
            extracted = self._extract_prediction_value(prediction)
            return self._parse_prediction(extracted) if extracted is not None else None

        if isinstance(prediction, list):
            if len(prediction) == 1:
                return self._parse_prediction(prediction[0])
            return None

        if isinstance(prediction, bool):
            candidate = int(prediction)
        elif isinstance(prediction, int):
            candidate = prediction
        elif isinstance(prediction, float) and prediction.is_integer():
            candidate = int(prediction)
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
                    return self._parse_prediction(parsed)

            answer_text = self._extract_answer_segment(text)
            matched = self._match_label_text(answer_text)
            if matched is not None:
                return matched

            lowered = answer_text.lower()
            if self.label_values_set == {0, 1}:
                if lowered in ("yes", "true"):
                    candidate = 1
                elif lowered in ("no", "false"):
                    candidate = 0
                else:
                    candidate = None
            else:
                candidate = None

            if candidate is None and self.label_value_type == "numeric":
                match = LABEL_PATTERN.search(answer_text)
                if not match:
                    return None
                candidate = int(match.group(0))
            elif candidate is None and self.label_value_type == "text":
                return None

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
