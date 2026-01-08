"""ASSIN2 STS - Semantic Textual Similarity in Portuguese."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..common import (
    FreeformHandler,
    CachedDatasetMixin,
    download_huggingface_dataset,
    save_to_jsonl,
    extract_float_score,
    format_score_with_comma,
    MeanSquaredError,
    PearsonCorrelation,
)

logger = logging.getLogger(__name__)


class Assin2Sts(CachedDatasetMixin, FreeformHandler):
    """ASSIN2 STS regression task - semantic similarity scoring (1.0 to 5.0)."""

    # Task configuration
    name = "assin2_sts"
    display_name = "ASSIN2 STS"
    description = (
        "Abaixo estão pares de frases que você deve avaliar o grau de similaridade. "
        "Dê uma pontuação entre 1,0 e 5,0, sendo 1,0 pouco similar e 5,0 muito similar.\n\n"
    )

    # Dataset configuration
    dataset_name = "assin2"
    split = "test"
    dataset_file = "assin2_sts_test.jsonl"

    # Score range for regression
    score_range = (1.0, 5.0)

    # Prompts
    system_prompt = ""

    # Metrics
    metrics = [MeanSquaredError(), PearsonCorrelation()]

    def _download_and_cache(self, output_path: Path):
        """
        Download the ASSIN2 STS dataset from HuggingFace, preprocess each sample, and write the processed samples to the given JSONL output path.
        
        Each processed sample is a dict with:
        - `id`: sample identifier (from `sentence_pair_id`, `id`, or the sample index).
        - `text`: two-line Portuguese prompt "Frase 1: {premise}\nFrase 2: {hypothesis}".
        - `expected`: relatedness score converted to `float` (from `relatedness_score`, `score`, or `0.0` if missing).
        
        Parameters:
            output_path (Path): Destination file path where the JSONL-formatted processed samples will be saved.
        """
        raw_samples = download_huggingface_dataset(
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=str(self.data_dir / "cache"),
        )

        processed = []
        for idx, raw_sample in enumerate(raw_samples):
            sample_id = raw_sample.get("sentence_pair_id", raw_sample.get("id", idx))
            score = raw_sample.get("relatedness_score", raw_sample.get("score", 0.0))

            premise = raw_sample.get("premise", "")
            hypothesis = raw_sample.get("hypothesis", "")
            text = f"Frase 1: {premise}\nFrase 2: {hypothesis}"

            processed.append({
                "id": str(sample_id),
                "text": text,
                "expected": float(score),
            })

        save_to_jsonl(processed, output_path)
        logger.info(f"Cached {len(processed)} samples to {output_path}")

    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Create the user prompt by concatenating the task description with the sample's text.
        
        Parameters:
            sample (Dict[str, Any]): Sample dictionary expected to contain a 'text' field; if absent, an empty string is used.
        
        Returns:
            str: Formatted prompt ready for the user, ending with "Pontuação:".
        """
        return f"{self.description}{sample.get('text', '')}\n\nPontuação:"

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute per-sample regression metrics by parsing a numeric score from the model prediction and comparing it to the expected score.
        
        Parameters:
            prediction: Model output (any type) that should contain a numeric score; it will be parsed into a float within the task's score_range.
            expected: The ground-truth score; should be convertible to float.
            sample (Dict): The input sample (not used for computation but provided for context).
            error (Optional[str]): Optional error message from the prediction pipeline; if present the metric is marked invalid.
            error_type (Optional[str]): Optional error type label to include in invalid results.
        
        Returns:
            dict: A dictionary with the following keys:
                - valid (bool): `True` if a numeric prediction and expected value were successfully parsed, `False` otherwise.
                - mse (float): Mean squared error between parsed prediction and expected value (0.0 for invalid results).
                - prediction (Optional[float]): Parsed predicted score when valid, otherwise `None`.
                - expected (Optional[float]): Parsed expected score when valid, otherwise `None`.
                - error (Optional[str]): Present when `valid` is `False`, describes the failure reason.
                - error_type (Optional[str]): Present when `valid` is `False`, classifies the failure.
        """
        if error or prediction is None:
            return {
                "valid": False,
                "error": error or "No prediction",
                "error_type": error_type,
                "mse": 0.0,
                "prediction": None,
                "expected": None,
            }

        # Extract float score from prediction
        pred_value = extract_float_score(
            str(prediction),
            min_value=self.score_range[0],
            max_value=self.score_range[1],
            fallback=None,
        )

        if pred_value is None:
            return {
                "valid": False,
                "error": "Could not parse score from response",
                "error_type": "invalid_response",
                "mse": 0.0,
                "prediction": None,
                "expected": None,
            }

        try:
            expected_value = float(expected)
        except (TypeError, ValueError):
            return {
                "valid": False,
                "error": "Invalid expected score",
                "error_type": "invalid_response",
                "mse": 0.0,
                "prediction": None,
                "expected": None,
            }

        mse = (pred_value - expected_value) ** 2
        return {
            "valid": True,
            "mse": mse,
            "prediction": pred_value,
            "expected": expected_value,
        }

    def aggregate_metrics(self, all_metrics: list) -> Dict[str, Any]:
        """
        Aggregate per-sample metric results into overall statistics.
        
        Returns:
            aggregated (Dict[str, Any]): Dictionary containing:
                - total_samples (int): total number of samples processed.
                - valid_samples (int): number of samples with valid metric results.
                - error_rate (float): fraction of samples that were invalid (0.0–1.0).
                - mse (float): mean squared error across valid samples; 0.0 if there are no valid samples.
                - pearson (float): Pearson correlation between predictions and expected values across valid samples; 0.0 if there is insufficient or degenerate data.
        """
        if not all_metrics:
            return {"total_samples": 0, "valid_samples": 0, "mse": 0.0, "pearson": 0.0}

        valid_metrics = [m for m in all_metrics if m.get("valid", False)]
        total_samples = len(all_metrics)
        valid_samples = len(valid_metrics)

        aggregated = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "error_rate": (total_samples - valid_samples) / total_samples if total_samples else 0.0,
        }

        if valid_samples > 0:
            # MSE
            mse_scores = [m.get("mse", 0.0) for m in valid_metrics]
            aggregated["mse"] = sum(mse_scores) / len(mse_scores)

            # Pearson correlation
            predictions = [m.get("prediction") for m in valid_metrics if m.get("prediction") is not None]
            expectations = [m.get("expected") for m in valid_metrics if m.get("expected") is not None]

            if len(predictions) >= 2 and len(predictions) == len(expectations):
                import math

                mean_pred = sum(predictions) / len(predictions)
                mean_exp = sum(expectations) / len(expectations)

                numerator = sum((p - mean_pred) * (e - mean_exp) for p, e in zip(predictions, expectations))
                denom_pred = sum((p - mean_pred) ** 2 for p in predictions)
                denom_exp = sum((e - mean_exp) ** 2 for e in expectations)
                denominator = math.sqrt(denom_pred * denom_exp)

                if denominator > 0:
                    aggregated["pearson"] = numerator / denominator
                else:
                    aggregated["pearson"] = 0.0
            else:
                aggregated["pearson"] = 0.0
        else:
            aggregated["mse"] = 0.0
            aggregated["pearson"] = 0.0

        return aggregated
