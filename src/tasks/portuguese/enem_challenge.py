"""ENEM Challenge - Brazilian National High School Exam Questions."""

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


class EnemChallenge(CachedDatasetMixin, MultipleChoiceHandler):
    """ENEM Challenge multiple-choice task."""

    # Task configuration
    name = "enem_challenge"
    display_name = "ENEM Challenge"
    description = (
        "As perguntas a seguir são questões de múltipla escolha do Exame Nacional do Ensino "
        "Médio (ENEM), selecione a única alternativa correta e responda apenas com as letras "
        "\"A\", \"B\", \"C\", \"D\" ou \"E\".\n\n"
    )

    # Dataset configuration
    dataset_name = "eduagarcia/enem_challenge"
    split = "train"
    dataset_file = "enem_test.jsonl"

    # Prompts
    system_prompt = ""

    def _download_and_cache(self, output_path: Path):
        """
        Download and preprocess the ENEM dataset and write the processed samples to output_path.
        
        Processes the dataset configured on the class (dataset_name and split), normalizes each sample into a dictionary with the keys: "id", "text", "choices", "choice_labels", and "expected", and saves the resulting list as a JSONL file.
        
        Parameters:
            output_path (Path): Destination file path for the cached JSONL containing processed samples.
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
        Extracts and normalizes choice labels and their corresponding texts from a raw dataset sample.
        
        Parameters:
            raw_sample (Dict): Raw sample expected to contain a "choices" field. The field may be a dict with label(s) and text(s) or a list of choice entries in varying shapes.
        
        Returns:
            tuple: (labels, texts) where `labels` is a list of choice labels as strings (e.g., "A", "B", ...) and `texts` is a list of corresponding choice texts as strings. If no labels are found, labels defaults to ["A", "B", "C", "D", "E"] truncated to the number of texts.
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
            labels = ["A", "B", "C", "D", "E"][: len(texts)]

        return labels, texts

    def _extract_expected_index(self, raw_sample: Dict, labels: list) -> int:
        """
        Determine the index of the correct choice for a raw sample.
        
        Parameters:
            raw_sample (Dict): Raw dataset sample; the answer key is read from "answerKey", or fallback to "answer" or "label".
            labels (list): Ordered list of choice labels (e.g., ["A", "B", "C"]).
        
        Returns:
            int: Index of the correct answer. If the answer key is an integer, it is returned directly. If it is a string matching an entry in `labels`, the index of that label is returned. If the answer key is missing or unrecognized, returns 0 and a warning is logged.
        """
        answer_key = raw_sample.get("answerKey", raw_sample.get("answer", raw_sample.get("label")))
        if isinstance(answer_key, int):
            return answer_key
        elif isinstance(answer_key, str) and answer_key in labels:
            return labels.index(answer_key)
        logger.warning("Missing answer key for sample, defaulting to 0")
        return 0

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Builds the task prompt combining the task description, question text, and formatted choices.
        
        Parameters:
            sample (Dict[str, Any]): Sample containing the question and choices. Expected keys:
                - 'text': the question text
                - 'choices': list of choice texts or choice objects
                - 'choice_labels': optional list of choice labels (e.g., ['A', 'B', ...])
        
        Returns:
            str: The full prompt string consisting of the task description, the question ("Pergunta:"), the formatted choices ("Alternativas:"), and the trailing "Resposta correta:" marker.
        """
        choices_text = format_choices(sample.get("choices", []), sample.get("choice_labels"))
        doc_text = (
            f"Pergunta:\n{sample.get('text', '')}\n"
            f"Alternativas:\n{choices_text}\n"
            "Resposta correta:"
        )
        return f"{self.description}{doc_text}".strip()
