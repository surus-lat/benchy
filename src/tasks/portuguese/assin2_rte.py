"""ASSIN2 RTE - Recognizing Textual Entailment in Portuguese."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..common import (
    FreeformHandler,
    CachedDatasetMixin,
    download_huggingface_dataset,
    save_to_jsonl,
    normalize_text,
)

logger = logging.getLogger(__name__)


class Assin2Rte(CachedDatasetMixin, FreeformHandler):
    """ASSIN2 RTE binary classification task (Sim/Não)."""

    # Task configuration
    name = "assin2_rte"
    display_name = "ASSIN2 RTE"
    description = (
        "Abaixo estão pares de premissa e hipótese. Para cada par, indique se a "
        "hipótese pode ser inferida a partir da premissa, responda apenas com \"Sim\" "
        "ou \"Não\".\n\n"
    )

    # Dataset configuration
    dataset_name = "assin2"
    split = "test"
    dataset_file = "assin2_rte_test.jsonl"

    # Labels for binary classification
    labels = ["Não", "Sim"]

    # Prompts
    system_prompt = ""

    def _download_and_cache(self, output_path: Path):
        """
        Download and preprocess the ASSIN2 RTE dataset and save the processed samples to the given path.
        
        Each processed sample is a dictionary with keys:
        - "id": string sample identifier (taken from "sentence_pair_id", "id", or the sample index)
        - "text": combined text in the form "Premissa: {premise}\nHipótese: {hypothesis}"
        - "expected": integer label where 0 = Não and 1 = Sim
        
        Parameters:
            output_path (Path): Destination JSONL file path where the cached processed samples will be written.
        """
        raw_samples = download_huggingface_dataset(
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=str(self.data_dir / "cache"),
        )

        processed = []
        for idx, raw_sample in enumerate(raw_samples):
            sample_id = raw_sample.get("sentence_pair_id", raw_sample.get("id", idx))
            entailment = raw_sample.get("entailment_judgment", raw_sample.get("label", 0))

            premise = raw_sample.get("premise", "")
            hypothesis = raw_sample.get("hypothesis", "")
            text = f"Premissa: {premise}\nHipótese: {hypothesis}"

            processed.append({
                "id": str(sample_id),
                "text": text,
                "expected": int(entailment),  # 0 = Não, 1 = Sim
            })

        save_to_jsonl(processed, output_path)
        logger.info(f"Cached {len(processed)} samples to {output_path}")

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Create the user-facing prompt by concatenating the task description, the sample's text, and a trailing "Resposta:".
        
        Parameters:
        	sample (Dict[str, Any]): Sample dictionary expected to contain a 'text' key with the formatted premise and hypothesis.
        
        Returns:
        	str: The complete prompt string: description + sample['text'] + two newlines followed by "Resposta:".
        """
        return f"{self.description}{sample.get('text', '')}\n\nResposta:"

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a model prediction against the expected binary label and return validation metrics.
        
        Parameters:
            prediction: The model-generated response to parse for a "Sim" or "Não" label.
            expected: The ground-truth label index (0 or 1) or a value convertible to int.
            sample: The original sample dictionary (used for context/diagnostics; not inspected here).
            error: Optional error message indicating an upstream failure; if present, the result is marked invalid.
            error_type: Optional string identifying the kind of error provided via `error`.
        
        Returns:
            A dictionary with:
                - "valid" (bool): `True` if the prediction was parsed and compared successfully, `False` otherwise.
                - "acc" (float): 1.0 if the predicted label index equals the expected index, 0.0 otherwise.
                - "predicted_idx" (int, optional): Index of the parsed prediction (0 for "Não", 1 for "Sim") when valid.
                - "expected_idx" (int, optional): Integer value of `expected` when valid.
                - "error" (str, optional): Error message when the result is invalid.
                - "error_type" (str, optional): Error type when the result is invalid.
        """
        if error or prediction is None:
            return {
                "valid": False,
                "error": error or "No prediction",
                "error_type": error_type,
                "acc": 0.0,
            }

        # Parse prediction
        predicted_label = self._extract_yes_no_label(str(prediction))
        if predicted_label is None:
            return {
                "valid": False,
                "error": "Could not parse Sim/Não from response",
                "error_type": "invalid_response",
                "acc": 0.0,
            }

        predicted_idx = self.labels.index(predicted_label)
        expected_idx = int(expected)

        return {
            "valid": True,
            "acc": 1.0 if predicted_idx == expected_idx else 0.0,
            "predicted_idx": predicted_idx,
            "expected_idx": expected_idx,
        }

    def _extract_yes_no_label(self, response: str) -> Optional[str]:
        """
        Extracts a "Sim" or "Não" label from a free-form text response.
        
        Searches the normalized text for standalone occurrences of "sim" or "nao" and returns the earliest match. If no standalone match is found, falls back to a substring check for "sim" then "nao". Returns None when neither label can be detected.
        
        Parameters:
            response (str): The text to parse for an affirmative or negative label.
        
        Returns:
            str | None: "Sim" if an affirmative label is detected, "Não" if a negative label is detected, or `None` if no label is found.
        """
        if not response:
            return None

        normalized = normalize_text(response)
        matches = []
        for label in ("sim", "nao"):
            import re
            match = re.search(rf"\b{label}\b", normalized)
            if match:
                matches.append((match.start(), label))

        if matches:
            matches.sort(key=lambda item: item[0])
            selected = matches[0][1]
        else:
            if "sim" in normalized:
                selected = "sim"
            elif "nao" in normalized:
                selected = "nao"
            else:
                return None

        return "Sim" if selected == "sim" else "Não"

    def aggregate_metrics(self, all_metrics: list) -> Dict[str, Any]:
        """
        Compute aggregated statistics from a list of per-sample metric dictionaries.
        
        Calculates the total number of samples, the number of valid samples (where `"valid"` is True), the error rate (proportion of invalid samples), and the average accuracy computed over valid samples.
        
        Parameters:
            all_metrics (list): List of per-sample metric dictionaries; each dictionary may include a boolean `"valid"` and a numeric `"acc"`.
        
        Returns:
            dict: Aggregated metrics with keys:
                total_samples (int): Total number of metric entries.
                valid_samples (int): Number of entries with `"valid"` == True.
                error_rate (float): Proportion of invalid entries (value between 0.0 and 1.0).
                acc (float): Mean of `"acc"` across valid entries, or 0.0 if there are no valid entries.
        """
        if not all_metrics:
            return {"total_samples": 0, "valid_samples": 0, "acc": 0.0}

        valid_metrics = [m for m in all_metrics if m.get("valid", False)]
        total_samples = len(all_metrics)
        valid_samples = len(valid_metrics)

        aggregated = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "error_rate": (total_samples - valid_samples) / total_samples if total_samples else 0.0,
        }

        if valid_samples > 0:
            acc_scores = [m.get("acc", 0.0) for m in valid_metrics]
            aggregated["acc"] = sum(acc_scores) / len(acc_scores)
        else:
            aggregated["acc"] = 0.0

        return aggregated
