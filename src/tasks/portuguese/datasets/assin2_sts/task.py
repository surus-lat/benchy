"""ASSIN2 STS task implementation."""

import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from ...base import PortugueseTaskBase
from ... import utils
from .....common.dataset_utils import download_huggingface_dataset, load_jsonl_dataset, save_to_jsonl

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / ".data" / "assin2_sts"
DEFAULT_DESCRIPTION = (
    "Abaixo estão pares de frases que você deve avaliar o grau de similaridade. "
    "Dê uma pontuação entre 1,0 e 5,0, sendo 1,0 pouco similar e 5,0 muito similar.\n\n"
)


class Assin2StsTask(PortugueseTaskBase):
    """ASSIN2 STS task."""

    @property
    def task_type(self) -> str:
        return "regression"

    @property
    def score_range(self) -> Tuple[float, float]:
        return (1.0, 5.0)

    def __init__(self, config: Dict):
        super().__init__(config)

        dataset_config = config.get("dataset", {})
        self.dataset_path = dataset_config.get("dataset_path", "assin2")
        self.split = dataset_config.get("split", "test")

        fewshot_config = config.get("fewshot", {})
        self.fewshot_split = fewshot_config.get("split", "train")
        self.fewshot_id_column = fewshot_config.get("id_column", "sentence_pair_id")
        self.fewshot_id_list = fewshot_config.get("id_list", [])
        self.num_fewshot = fewshot_config.get("num_fewshot", config.get("num_fewshot", 15))
        self.exclude_from_task = fewshot_config.get("exclude_from_task", False)

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
            score = raw_sample.get("relatedness_score", raw_sample.get("score", 0.0))
            processed.append(
                {
                    "id": str(sample_id),
                    "premise": raw_sample.get("premise", ""),
                    "hypothesis": raw_sample.get("hypothesis", ""),
                    "expected": float(score),
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
            score = raw_sample.get("relatedness_score", raw_sample.get("score", 0.0))
            selected.append(
                {
                    "premise": raw_sample.get("premise", ""),
                    "hypothesis": raw_sample.get("hypothesis", ""),
                    "score": float(score),
                }
            )

        save_to_jsonl(selected, self.fewshot_file)

    def _load_fewshot_prompt(self) -> str:
        if not self.fewshot_file.exists():
            return ""

        fewshot_samples = load_jsonl_dataset(self.fewshot_file)
        formatted = [self._format_fewshot_example(sample) for sample in fewshot_samples]
        return utils.build_fewshot_block(formatted)

    def _format_fewshot_example(self, sample: Dict) -> str:
        score = utils.format_score_pt(sample.get("score", 0.0))
        text = (
            f"Frase 1: {sample.get('premise', '')}\n"
            f"Frase 2: {sample.get('hypothesis', '')}\n"
            "Pergunta: Quão similares são as duas frases? Dê uma pontuação entre 1,0 a 5,0.\n"
            "Resposta:"
        )
        return f"{text} {score}".strip()

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
            f"Frase 1: {sample.get('premise', '')}\n"
            f"Frase 2: {sample.get('hypothesis', '')}\n"
            "Pergunta: Quão similares são as duas frases? Dê uma pontuação entre 1,0 a 5,0.\n"
            "Resposta:"
        )
        user_prompt = f"{self.description}{self.fewshot_prompt}{doc_text}".strip()
        return system_prompt, user_prompt

    def get_task_name(self) -> str:
        return "assin2_sts"
