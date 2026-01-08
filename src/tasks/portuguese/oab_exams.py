"""OAB Exams - Brazilian Bar Association Exam Questions."""

import logging
from pathlib import Path
from typing import Dict, Any

from ..common import (
    MultipleChoiceHandler,
    CachedDatasetMixin,
    download_huggingface_dataset,
    save_to_jsonl,
    format_choices,
)

logger = logging.getLogger(__name__)


class OabExams(CachedDatasetMixin, MultipleChoiceHandler):
    """OAB Exams multiple-choice task."""

    # Task configuration
    name = "oab_exams"
    display_name = "OAB Exams"
    description = (
        "As perguntas a seguir são questões de múltipla escolha do Exame de Ordem da Ordem "
        "dos Advogados do Brasil (OAB), selecione a única alternativa correta e responda "
        "apenas com as letras \"A\", \"B\", \"C\" ou \"D\".\n\n"
    )

    # Dataset configuration
    dataset_name = "eduagarcia/oab_exams"
    split = "train"
    dataset_file = "oab_test.jsonl"

    # Prompts
    system_prompt = ""

    def _download_and_cache(self, output_path: Path):
        """Download and preprocess OAB dataset."""
        raw_samples = download_huggingface_dataset(
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=str(self.data_dir / "cache"),
        )

        processed = []
        for idx, raw_sample in enumerate(raw_samples):
            sample_id = raw_sample.get("id", idx)
            labels, choices = self._extract_choices(raw_sample)
            expected_idx = self._extract_expected_index(raw_sample, labels)

            processed.append({
                "id": str(sample_id),
                "text": raw_sample.get("question", raw_sample.get("text", "")),
                "choices": choices,
                "choice_labels": labels,
                "expected": expected_idx,
            })

        save_to_jsonl(processed, output_path)
        logger.info(f"Cached {len(processed)} samples to {output_path}")

    def _extract_choices(self, raw_sample: Dict) -> tuple:
        """Extract choice labels and texts from raw sample."""
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
            labels = ["A", "B", "C", "D"][: len(texts)]

        return labels, texts

    def _extract_expected_index(self, raw_sample: Dict, labels: list) -> int:
        """Extract the expected answer index."""
        answer_key = raw_sample.get("answerKey", raw_sample.get("answer", raw_sample.get("label")))
        if isinstance(answer_key, int):
            return answer_key
        elif isinstance(answer_key, str) and answer_key in labels:
            return labels.index(answer_key)
        logger.warning(f"Missing answer key for sample, defaulting to 0")
        return 0

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """Format the user prompt for a sample."""
        choices_text = format_choices(sample.get("choices", []), sample.get("choice_labels"))
        doc_text = (
            f"Pergunta:\n{sample.get('text', '')}\n"
            f"Alternativas:\n{choices_text}\n"
            "Resposta correta:"
        )
        return f"{self.description}{doc_text}".strip()

