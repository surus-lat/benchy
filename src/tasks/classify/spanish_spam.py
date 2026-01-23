"""Spanish spam/ham classification subtask.

Binary classification task for detecting spam vs ham in Spanish messages.
Dataset: softecapps/spam_ham_spanish
CLI alias: classify.spanish-spam
"""

import logging
import csv
from typing import Any, Dict, List, Optional

from ..common import MultipleChoiceHandler
from ..common.utils.dataset_utils import load_jsonl_dataset, save_to_jsonl

logger = logging.getLogger(__name__)


class SpanishSpam(MultipleChoiceHandler):
    """Spanish spam/ham detection (binary classification)."""

    name = "spanish_spam"
    display_name = "spanish-spam"
    description = "Binary classification for Spanish spam vs ham detection"

    dataset = "softecapps/spam_ham_spanish"
    split = "test"
    default_data_file = "spanish_spam.jsonl"

    system_prompt = "Eres un asistente que clasifica mensajes."

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from HF `test.csv`, tolerating broken split generation.

        The upstream dataset currently fails with `datasets` due to mismatched CSV
        headers across files (e.g. "tipo" vs " tipo"). We avoid `load_dataset()`
        and download the CSV directly, normalizing column names by stripping
        whitespace.
        """
        if self.data_file.exists():
            return load_jsonl_dataset(self.data_file)

        def _download_csv(filename: str) -> str:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(
                repo_id=self.dataset,
                repo_type="dataset",
                filename=filename,
                cache_dir=str(self.data_dir / "hf_cache"),
            )

        csv_path: Optional[str] = None
        last_error: Optional[Exception] = None
        for candidate in ("test.csv", "train.csv"):
            try:
                csv_path = _download_csv(candidate)
                break
            except Exception as e:
                last_error = e
                continue

        if not csv_path:
            raise RuntimeError(f"Failed to download CSV for {self.dataset}: {last_error}")

        processed: List[Dict[str, Any]] = []
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, raw_row in enumerate(reader):
                # Normalize keys to avoid issues like " tipo" vs "tipo"
                row = {(k.strip() if isinstance(k, str) else k): v for k, v in (raw_row or {}).items()}
                sample = self.preprocess_sample(row, idx)
                if sample is not None:
                    processed.append(sample)

        save_to_jsonl(processed, self.data_file)
        return processed

    def preprocess_sample(
        self, raw_sample: Dict[str, Any], idx: int
    ) -> Optional[Dict[str, Any]]:
        """Map dataset fields:
        - mensaje: input text
        - tipo: 'spam' or 'ham'
        """
        text = raw_sample.get("mensaje")
        label = raw_sample.get("tipo")

        if text is None or label is None:
            return None

        label_str = str(label).strip().lower()
        if label_str not in {"spam", "ham"}:
            return None

        choices = ["ham", "spam"]
        label_to_index = {"ham": 0, "spam": 1}

        return {
            "id": f"{self.get_task_name()}_{idx}",
            "text": str(text),
            "expected": label_to_index[label_str],
            "choices": choices,
            "choice_labels": ["ham", "spam"],
            "label_to_index": label_to_index,
            "label_value": label_str,
        }

    def get_prompt(self, sample: Dict[str, Any]):
        user_prompt = (
            "Clasifica el siguiente mensaje como spam o ham (legitimo).\n\n"
            f"Mensaje:\n{sample.get('text', '')}\n\n"
            "Responde únicamente con una de estas etiquetas: ham, spam."
        )
        return self.system_prompt, user_prompt
