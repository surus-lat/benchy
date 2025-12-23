"""FaQuAD-NLI task implementation."""

import csv
import json
import logging
import gzip
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Any, List

from huggingface_hub import hf_hub_download

from ...base import PortugueseTaskBase
from ... import utils
from .....common.dataset_utils import load_jsonl_dataset, save_to_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / ".data" / "faquad_nli"
DEFAULT_DESCRIPTION = (
    "Abaixo estão pares de pergunta e resposta. Para cada par, você deve julgar se a "
    "resposta responde à pergunta de maneira satisfatória e aparenta estar correta. "
    "Escreva apenas \"Sim\" ou \"Não\".\n\n"
)

DEFAULT_FEWSHOT_INDICES = [
    1893, 949, 663, 105, 1169, 2910, 2227, 2813, 974, 558,
    1503, 1958, 2918, 601, 1560, 984, 2388, 995, 2233, 1982,
    165, 2788, 1312, 2285, 522, 1113, 1670, 323, 236, 1263,
    1562, 2519, 1049, 432, 1167, 1394, 2022, 2551, 2194, 2187,
    2282, 2816, 108, 301, 1185, 1315, 1420, 2436, 2322, 766,
]


class FaquadNliTask(PortugueseTaskBase):
    """FaQuAD-NLI classification task."""

    @property
    def task_type(self) -> str:
        return "classification"

    @property
    def labels(self):
        return ["Não", "Sim"]

    def __init__(self, config: Dict):
        super().__init__(config)

        dataset_config = config.get("dataset", {})
        self.dataset_path = dataset_config.get("dataset_path", "ruanchaves/faquad-nli")
        self.split = dataset_config.get("split", "test")
        self.data_filename = dataset_config.get("data_filename", "spans.csv")

        fewshot_config = config.get("fewshot", {})
        self.fewshot_split = fewshot_config.get("split", "train")
        self.fewshot_indices = fewshot_config.get("indices", DEFAULT_FEWSHOT_INDICES)
        self.num_fewshot = fewshot_config.get("num_fewshot", config.get("num_fewshot", 15))

        self.description = config.get("description", DEFAULT_DESCRIPTION)

        self.data_file = DATA_DIR / f"{self.split}.jsonl"
        self.fewshot_file = DATA_DIR / "fewshot.jsonl"

        self.dataset: Optional[list] = None
        self.fewshot_prompt: str = ""

    def load(self) -> None:
        if not self.data_file.exists() or (self.fewshot_indices and not self.fewshot_file.exists()):
            self._download_and_preprocess()

        self.dataset = load_jsonl_dataset(self.data_file)
        self.fewshot_prompt = self._load_fewshot_prompt()

    def _download_and_preprocess(self) -> None:
        raw_rows = self._load_raw_rows()
        split_rows = self._filter_rows_by_split(raw_rows, self.split)

        processed = []
        skipped = 0
        for idx, row in enumerate(split_rows):
            sample = self._row_to_sample(row, idx)
            if not sample:
                skipped += 1
                continue
            processed.append(sample)

        if skipped:
            logger.warning(f"Skipped {skipped} rows missing required fields")

        save_to_jsonl(processed, self.data_file)

        if self.fewshot_indices:
            self._save_fewshot_samples(raw_rows)

    def _save_fewshot_samples(self, raw_rows: List[Dict[str, Any]]) -> None:
        fewshot_rows = self._filter_rows_by_split(raw_rows, self.fewshot_split)
        selected = []
        for row_idx in self.fewshot_indices[: self.num_fewshot]:
            if row_idx >= len(fewshot_rows):
                logger.warning(f"Few-shot index out of range: {row_idx}")
                continue
            row = fewshot_rows[row_idx]
            sample = self._row_to_sample(row, row_idx)
            if not sample:
                continue
            selected.append(
                {
                    "question": sample.get("question", ""),
                    "answer": sample.get("answer", ""),
                    "label": self.labels[int(sample.get("expected_idx", 0))],
                }
            )

        save_to_jsonl(selected, self.fewshot_file)

    def _load_fewshot_prompt(self) -> str:
        if not self.fewshot_file.exists():
            return ""

        fewshot_samples = load_jsonl_dataset(self.fewshot_file)
        formatted = [self._format_fewshot_example(sample) for sample in fewshot_samples]
        return utils.build_fewshot_block(formatted)

    def _download_spans_csv(self) -> Path:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache_dir = str(DATA_DIR / "cache")
        candidates = [self.data_filename, "spans.csv", "data/spans.csv", "spans.tsv", "data/spans.tsv", "spans.csv.gz"]

        last_error = None
        for filename in candidates:
            try:
                return Path(
                    hf_hub_download(
                        repo_id=self.dataset_path,
                        filename=filename,
                        repo_type="dataset",
                        cache_dir=cache_dir,
                    )
                )
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(f"Unable to download spans file from {self.dataset_path}: {last_error}")

    def _load_raw_rows(self) -> List[Dict[str, Any]]:
        csv_path = self._download_spans_csv()
        open_fn = gzip.open if csv_path.suffix == ".gz" else open
        delimiter = "\t" if ".tsv" in csv_path.suffixes else ","
        rows = []
        with open_fn(csv_path, "rt", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                rows.append(row)
        return rows

    def _filter_rows_by_split(self, rows: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
        if not rows:
            return rows

        split_lower = split.lower()
        split_columns = ["split", "subset", "partition"]
        for column in split_columns:
            if column in rows[0]:
                filtered = [row for row in rows if str(row.get(column, "")).lower() == split_lower]
                if not filtered:
                    logger.warning(f"No rows matched split '{split}' in column '{column}', using all rows")
                    return rows
                return filtered

        flag_map = {
            "train": "is_train",
            "test": "is_test",
            "validation": "is_validation",
            "valid": "is_validation",
            "dev": "is_validation",
        }
        flag_column = flag_map.get(split_lower)
        if flag_column and flag_column in rows[0]:
            filtered = [row for row in rows if str(row.get(flag_column, "")).lower() in {"1", "true", "yes"}]
            if not filtered:
                logger.warning(f"No rows matched split '{split}' via flag '{flag_column}', using all rows")
                return rows
            return filtered

        return rows

    def _row_to_sample(self, row: Dict[str, Any], fallback_id: int) -> Optional[Dict[str, Any]]:
        question = self._get_first_value(row, ["question", "question_text", "pergunta", "query", "questao"])
        answer_raw = self._get_first_value(row, ["answer", "answer_text", "resposta", "span", "answer_span"])
        label_raw = self._get_first_value(
            row,
            ["label", "gold_label", "entailment", "is_correct", "answerable", "is_answerable"],
        )

        if not question or answer_raw is None:
            return None

        label = self._parse_label(label_raw)
        if label is None:
            label = 0

        sample_id = self._get_first_value(row, ["id", "qid", "question_id", "uuid"])
        answer = self._normalize_answer_field(answer_raw)

        return {
            "id": str(sample_id or fallback_id),
            "question": question,
            "answer": answer,
            "expected": int(label),
            "expected_idx": int(label),
        }

    def _get_first_value(self, row: Dict[str, Any], keys: List[str]) -> Optional[str]:
        for key in keys:
            if key in row and row[key] not in (None, ""):
                return row[key]
        return None

    def _normalize_answer_field(self, answer_field: Any) -> str:
        if answer_field is None:
            return ""
        if isinstance(answer_field, list):
            return str(answer_field[0]) if answer_field else ""
        if isinstance(answer_field, dict):
            return str(answer_field.get("text") or answer_field.get("answer") or "")

        answer_text = str(answer_field).strip()
        if answer_text.startswith("{") or answer_text.startswith("["):
            try:
                parsed = json.loads(answer_text)
                return self._normalize_answer_field(parsed)
            except json.JSONDecodeError:
                return answer_text
        return answer_text

    def _parse_label(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)

        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "sim", "entailment", "correct"}:
            return 1
        if text in {"0", "false", "no", "nao", "não", "contradiction", "incorrect"}:
            return 0

        try:
            numeric = int(float(text))
            return 1 if numeric >= 1 else 0
        except ValueError:
            return None

    def _format_fewshot_example(self, sample: Dict) -> str:
        text = (
            f"Pergunta: {sample.get('question', '')}\n"
            f"Resposta: {sample.get('answer', '')}\n"
            "A resposta dada satisfaz à pergunta? Sim ou Não?"
        )
        return f"{text}\n{sample.get('label', '')}".strip()

    def get_samples(self, limit: Optional[int] = None) -> Iterator[Dict]:
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        data = self.dataset
        if limit is not None:
            data = data[: min(limit, len(self.dataset))]

        for sample in data:
            yield sample

    def get_prompt(self, sample: Dict) -> Tuple[str, str]:
        system_prompt = self.config.get("prompts", {}).get("system", "")
        doc_text = (
            f"Pergunta: {sample.get('question', '')}\n"
            f"Resposta: {sample.get('answer', '')}\n"
            "A resposta dada satisfaz à pergunta? Sim ou Não?"
        )
        user_prompt = f"{self.description}{self.fewshot_prompt}{doc_text}".strip()
        return system_prompt, user_prompt

    def get_task_name(self) -> str:
        return "faquad_nli"
