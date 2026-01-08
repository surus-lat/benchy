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
        """
        Download the configured OAB dataset from HuggingFace, preprocess each sample, and save the processed samples as a JSONL file.
        
        Each processed sample will include the keys: `id`, `text`, `choices`, `choice_labels`, and `expected` (index of the correct choice). The dataset is loaded from the class's `dataset_name` and `split`, and a local cache directory under `self.data_dir / "cache"` is used for the raw download.
        
        Parameters:
            output_path (Path): Filesystem path where the resulting JSONL file of processed samples will be written.
        """
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
        """
        Extracts choice labels and option texts from a raw dataset sample.
        
        Supports 'choices' as a dict (with "label"/"labels" and "text"/"texts") or as a list of dicts or values. Normalizes and trims values to lists of strings; if no labels are present, returns default labels "A", "B", "C", ... up to the number of texts.
        
        Parameters:
            raw_sample (Dict): Raw dataset sample expected to contain a "choices" field.
        
        Returns:
            tuple: (labels, texts) where `labels` is a list of label strings and `texts` is a list of option text strings.
        """
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
        """
        Determine the index of the correct answer within the provided labels.
        
        Parameters:
            raw_sample (Dict): A sample dictionary that may contain the correct answer under the keys
                "answerKey", "answer", or "label".
            labels (list): Ordered list of choice labels (for example ["A", "B", "C", "D"]).
        
        Returns:
            int: The index of the correct choice:
                - If the sample's answer is an integer, that integer is returned.
                - If the sample's answer is a string and matches an entry in `labels`, the matching index is returned.
                - Returns 0 if no valid answer key is found or the answer does not match `labels`.
        """
        answer_key = raw_sample.get("answerKey", raw_sample.get("answer", raw_sample.get("label")))
        if isinstance(answer_key, int):
            return answer_key
        elif isinstance(answer_key, str) and answer_key in labels:
            return labels.index(answer_key)
        logger.warning(f"Missing answer key for sample, defaulting to 0")
        return 0

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Builds the full user prompt for a sample by combining the task description, question text, and formatted choices.
        
        Parameters:
            sample (Dict[str, Any]): A mapping representing the sample. Expected keys include:
                - "text": the question text.
                - "choices": the raw choices data used by the choice formatter.
                - "choice_labels": optional labels for each choice.
        
        Returns:
            str: The assembled prompt containing the class description, the question ("Pergunta:"), the formatted alternatives ("Alternativas:"), and the trailing "Resposta correta:" marker.
        """
        choices_text = format_choices(sample.get("choices", []), sample.get("choice_labels"))
        doc_text = (
            f"Pergunta:\n{sample.get('text', '')}\n"
            f"Alternativas:\n{choices_text}\n"
            "Resposta correta:"
        )
        return f"{self.description}{doc_text}".strip()
