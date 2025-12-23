"""OAB Exams task implementation."""

import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from ...base import PortugueseTaskBase
from ... import utils
from .....common.dataset_utils import download_huggingface_dataset, load_jsonl_dataset, save_to_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / ".data" / "oab_exams"
DEFAULT_DESCRIPTION = (
    "As perguntas a seguir são questões de múltipla escolha do Exame de Ordem da Ordem "
    "dos Advogados do Brasil (OAB), selecione a única alternativa correta e responda "
    "apenas com as letras \"A\", \"B\", \"C\" ou \"D\".\n\n"
)

DEFAULT_FEWSHOT_IDS = [
    "2010-01_1", "2010-01_11", "2010-01_13", "2010-01_23", "2010-01_26",
    "2010-01_28", "2010-01_38", "2010-01_48", "2010-01_58", "2010-01_68",
    "2010-01_76", "2010-01_83", "2010-01_85", "2010-01_91", "2010-01_99",
]


class OabExamsTask(PortugueseTaskBase):
    """OAB Exams multiple-choice task."""

    @property
    def task_type(self) -> str:
        return "multiple_choice"

    def __init__(self, config: Dict):
        super().__init__(config)

        dataset_config = config.get("dataset", {})
        self.dataset_path = dataset_config.get("dataset_path", "eduagarcia/oab_exams")
        self.split = dataset_config.get("split", "train")

        fewshot_config = config.get("fewshot", {})
        self.fewshot_split = fewshot_config.get("split", "train")
        self.fewshot_id_column = fewshot_config.get("id_column", "id")
        self.fewshot_id_list = fewshot_config.get("id_list", DEFAULT_FEWSHOT_IDS)
        self.num_fewshot = fewshot_config.get("num_fewshot", config.get("num_fewshot", 3))
        self.exclude_from_task = fewshot_config.get("exclude_from_task", True)

        self.description = config.get("description", DEFAULT_DESCRIPTION)

        self.data_file = DATA_DIR / f"{self.split}.jsonl"
        self.fewshot_file = DATA_DIR / "fewshot.jsonl"

        self.dataset: Optional[list] = None
        self.fewshot_prompt: str = ""

    def load(self) -> None:
        if not self.data_file.exists() or (self.fewshot_id_list and not self.fewshot_file.exists()):
            self._download_and_preprocess()

        self.dataset = load_jsonl_dataset(self.data_file)
        self.fewshot_prompt = self._load_fewshot_prompt()

    def _download_and_preprocess(self) -> None:
        raw_samples = download_huggingface_dataset(
            dataset_name=self.dataset_path,
            split=self.split,
            cache_dir=str(DATA_DIR / "cache"),
        )

        processed = []
        for idx, raw_sample in enumerate(raw_samples):
            sample_id = raw_sample.get(self.fewshot_id_column, raw_sample.get("id", idx))
            labels, choices = self._extract_choices(raw_sample)
            expected_idx = self._extract_expected_index(raw_sample, labels)

            processed.append(
                {
                    "id": str(sample_id),
                    "question": raw_sample.get("question", raw_sample.get("text", "")),
                    "choices": choices,
                    "choice_labels": labels,
                    "expected": expected_idx,
                    "expected_idx": expected_idx,
                    "exam_id": raw_sample.get("exam_id"),
                }
            )

        if self.exclude_from_task and self.fewshot_id_list:
            exclude_set = {str(item) for item in self.fewshot_id_list[: self.num_fewshot]}
            processed = [sample for sample in processed if sample["id"] not in exclude_set]

        save_to_jsonl(processed, self.data_file)

        if self.fewshot_id_list:
            self._save_fewshot_samples()

    def _save_fewshot_samples(self) -> None:
        fewshot_raw = download_huggingface_dataset(
            dataset_name=self.dataset_path,
            split=self.fewshot_split,
            cache_dir=str(DATA_DIR / "cache"),
        )

        id_map = {}
        for idx, raw_sample in enumerate(fewshot_raw):
            sample_id = raw_sample.get(self.fewshot_id_column, raw_sample.get("id", idx))
            id_map[str(sample_id)] = raw_sample

        selected = []
        for sample_id in self.fewshot_id_list[: self.num_fewshot]:
            raw_sample = id_map.get(str(sample_id))
            if not raw_sample:
                logger.warning(f"Few-shot id not found: {sample_id}")
                continue
            labels, choices = self._extract_choices(raw_sample)
            expected_idx = self._extract_expected_index(raw_sample, labels)
            answer_label = labels[expected_idx] if expected_idx is not None else ""
            selected.append(
                {
                    "question": raw_sample.get("question", raw_sample.get("text", "")),
                    "choices": choices,
                    "choice_labels": labels,
                    "answer": answer_label,
                }
            )

        save_to_jsonl(selected, self.fewshot_file)

    def _load_fewshot_prompt(self) -> str:
        if not self.fewshot_file.exists():
            return ""

        fewshot_samples = load_jsonl_dataset(self.fewshot_file)
        formatted = [self._format_fewshot_example(sample) for sample in fewshot_samples]
        return utils.build_fewshot_block(formatted)

    def _extract_choices(self, raw_sample: Dict) -> Tuple[list, list]:
        choices = raw_sample.get("choices", {})
        labels = []
        texts = []

        if isinstance(choices, dict):
            labels = choices.get("label") or choices.get("labels") or []
            texts = choices.get("text") or choices.get("texts") or []
        elif isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict):
                    labels.append(choice.get("label") or choice.get("key") or choice.get("id"))
                    texts.append(choice.get("text") or choice.get("content") or choice.get("option"))
                else:
                    texts.append(str(choice))

        labels = [str(label).strip() for label in labels if label is not None]
        texts = [str(text).strip() for text in texts if text is not None]

        if not labels:
            labels = utils.CHOICE_LABELS[: len(texts)]

        return labels, texts

    def _extract_expected_index(self, raw_sample: Dict, labels: list) -> int:
        answer_key = raw_sample.get("answerKey", raw_sample.get("answer", raw_sample.get("label")))
        if isinstance(answer_key, int):
            expected_idx = answer_key
        elif isinstance(answer_key, str):
            expected_idx = labels.index(answer_key) if answer_key in labels else None
        else:
            expected_idx = None

        if expected_idx is None:
            logger.warning("Missing answer key; defaulting to first option")
            expected_idx = 0

        return expected_idx

    def _format_fewshot_example(self, sample: Dict) -> str:
        choices_text = utils.format_choices(sample.get("choices", []), sample.get("choice_labels"))
        text = (
            f"Questão:\n{sample.get('question', '')}\n"
            f"Alternativas:\n{choices_text}\n"
            "Resposta correta:"
        )
        return f"{text} {sample.get('answer', '')}".strip()

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
        choices_text = utils.format_choices(sample.get("choices", []), sample.get("choice_labels"))
        doc_text = (
            f"Questão:\n{sample.get('question', '')}\n"
            f"Alternativas:\n{choices_text}\n"
            "Resposta correta:"
        )
        user_prompt = f"{self.description}{self.fewshot_prompt}{doc_text}".strip()
        return system_prompt, user_prompt

    def get_task_name(self) -> str:
        return "oab_exams"
