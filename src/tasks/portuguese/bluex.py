"""BLUEX - Brazilian University Entrance Exam Questions."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..common import (
    MultipleChoiceHandler,
    CachedDatasetMixin,
    download_huggingface_dataset,
    save_to_jsonl,
    load_jsonl_dataset,
)

logger = logging.getLogger(__name__)


class Bluex(CachedDatasetMixin, MultipleChoiceHandler):
    """BLUEX multiple-choice task - Brazilian university entrance exam questions."""

    # Task configuration
    name = "bluex"
    display_name = "BLUEX"
    description = (
        "As perguntas a seguir são questões de múltipla escolha de provas de vestibular "
        "de universidades brasileiras, selecione a única alternativa correta e responda "
        "apenas com as letras \"A\", \"B\", \"C\", \"D\" ou \"E\".\n\n"
    )

    # Dataset configuration
    dataset_name = "eduagarcia-temp/BLUEX_without_images"
    split = "train"
    dataset_file = "bluex_test.jsonl"

    # Prompts
    system_prompt = ""

    def _download_and_cache(self, output_path: Path):
        """
        Download the configured BLUEX dataset, normalize each sample, and write a JSONL cache to the given path.
        
        Each cached record contains the fields: `id` (string), `text` (question text), `choices` (list of choice texts), `choice_labels` (list of choice labels), and `expected` (index of the correct choice). The dataset source and split are taken from the instance configuration.
        
        Parameters:
            output_path (Path): Path to write the resulting JSONL file.
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
        Normalize and extract choice labels and their corresponding texts from a raw dataset sample.
        
        Parameters:
            raw_sample (Dict): Raw sample dictionary expected to contain a "choices" field, which may be a mapping or a list of choice entries.
        
        Returns:
            tuple: (labels, texts) where `labels` is a list of choice label strings (e.g., "A", "B", ...) and `texts` is a list of choice text strings. Both lists are aligned by index.
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
        Resolve the index of the correct answer for a raw sample.
        
        Checks the keys "answerKey", "answer", and "label" (in that order). If the answer is an integer, it is returned as-is. If it is a string that matches one of `labels`, the corresponding index is returned. If no valid answer is found, a warning is logged and `0` is returned.
        
        Parameters:
            raw_sample (Dict): The raw dataset sample which may contain the answer under "answerKey", "answer", or "label".
            labels (list): Ordered list of choice labels to map label strings to indices.
        
        Returns:
            int: Index of the correct choice; `0` if the answer key is missing or cannot be resolved.
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
        Builds the user-facing prompt for a BLUEX sample in Portuguese.
        
        Constructs a prompt containing the question text, the formatted list of alternatives, and the cue "Resposta correta:" appended at the end. The module-level `description` is prefixed to the prompt.
        
        Parameters:
            sample (Dict[str, Any]): Sample dictionary expected to contain at least:
                - "text": question text,
                - "choices": list of choice texts,
                - "choice_labels" (optional): list of labels for the choices.
        
        Returns:
            str: The full formatted prompt string ready to present to the model.
        """
        from ..common import format_choices

        choices_text = format_choices(sample.get("choices", []), sample.get("choice_labels"))
        doc_text = (
            f"Pergunta:\n{sample.get('text', '')}\n"
            f"Alternativas:\n{choices_text}\n"
            "Resposta correta:"
        )
        return f"{self.description}{doc_text}".strip()
