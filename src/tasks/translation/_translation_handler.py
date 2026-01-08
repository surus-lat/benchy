"""Translation handler base class.

Extends FreeformHandler with translation-specific features:
- COMET model management (passed via config)
- Bidirectional language pair aggregation
- Translation metrics (BLEU, chrF, COMET)
"""

import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

from ..common import FreeformHandler
from .metrics import TranslationMetricsCalculator

logger = logging.getLogger(__name__)


class TranslationHandler(FreeformHandler):
    """Base handler for translation tasks.
    
    This handler extends FreeformHandler with:
    - COMET model management (shared across subtasks)
    - Translation metrics (BLEU, chrF, COMET)
    - Bidirectional pair aggregation (A->B and B->A averaged)
    - Per-language and per-pair metric breakdowns
    
    COMET Model Lifecycle:
    ----------------------
    The COMET model is expensive to load (2-5 minutes, 1.5GB).
    It should be loaded ONCE per task group via TaskGroupSpec setup:
    
    ```python
    TRANSLATION_SPEC = TaskGroupSpec(
        name="translation",
        setup=lambda ctx: {"comet_model": load_comet_model()},
        ...
    )
    ```
    
    The model is passed via config and shared across all subtasks.
    
    Subclasses should implement:
    - load_dataset() - Load preprocessed bidirectional pairs
    - get_prompt() - Build translation prompt
    
    Data Format:
    ------------
    Each sample should have:
    - id: unique identifier
    - source_text: text to translate
    - target_text: reference translation (expected)
    - source_language: source language name
    - target_language: target language name
    - language_pair: pair identifier (e.g., "eng_spa")
    - direction: translation direction (e.g., "eng->spa")
    """
    
    # Task type
    answer_type = "freeform"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Create and configure a TranslationHandler instance with optional shared COMET model.
        
        Parameters:
            config (Optional[Dict[str, Any]]): Optional configuration dictionary. Recognized keys:
                - "comet_model": a preloaded COMET model instance intended to be shared across subtasks.
                - "language_pairs": list of language-pair identifiers to evaluate.
        """
        super().__init__(config)
        
        # COMET model (passed from setup)
        self.comet_model = self.config.get("comet_model") if self.config else None
        
        # Initialize metrics calculator with COMET model
        self._metrics_calc = None
    
    @property
    def metrics_calculator(self) -> TranslationMetricsCalculator:
        """
        Provide a cached TranslationMetricsCalculator configured with this handler's COMET model.
        
        Returns:
            TranslationMetricsCalculator: The metrics calculator instance; created on first access and reused thereafter.
        """
        if self._metrics_calc is None:
            self._metrics_calc = TranslationMetricsCalculator({
                "comet_model": self.comet_model
            })
        return self._metrics_calc
    
    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute translation evaluation metrics for a single sample and attach language metadata.
        
        Parameters:
        	sample (Dict): Full sample dictionary; used to extract `language_pair`, `direction`, `source_language`, and `target_language`.
        
        Returns:
        	metrics (Dict[str, Any]): Evaluation results including metric scores (e.g., `bleu`, `chrf`, `comet` when available), status flags such as `valid`, and the metadata keys `language_pair`, `direction`, `source_language`, and `target_language`.
        """
        # Add language pair info to metrics for aggregation
        metrics = self.metrics_calculator.calculate(
            prediction=prediction,
            expected=expected,
            error=error,
            error_type=error_type,
        )
        
        # Add metadata for aggregation
        metrics["language_pair"] = sample.get("language_pair", "unknown")
        metrics["direction"] = sample.get("direction", "unknown")
        metrics["source_language"] = sample.get("source_language", "unknown")
        metrics["target_language"] = sample.get("target_language", "unknown")
        
        return metrics
    
    def get_error_metrics(
        self,
        error: str,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Return metrics for a prediction that failed due to an error.
        
        Parameters:
            error (str): Error message describing the failure.
            error_type (Optional[str]): Categorized error type, if available.
        
        Returns:
            dict: Metrics dictionary for the error case, including error details and metric fields populated for failed predictions.
        """
        return self.metrics_calculator.calculate(
            prediction=None,
            expected="",
            error=error,
            error_type=error_type,
        )
    
    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate translation metrics across a list of per-sample metric dictionaries.
        
        Parameters:
            all_metrics (List[Dict]): Per-sample metric objects produced by calculate_metrics (or similarly shaped),
                where each entry may include keys like `valid`, `language_pair`, `direction`, `comet`, `bleu`, and `chrf`.
        
        Returns:
            Dict[str, Any]: Summary dictionary containing:
                - total_samples (int): Number of input metric entries.
                - valid_samples (int): Count of entries marked as valid.
                - error_count (int): Count of entries not marked as valid.
                - error_rate (float): error_count / total_samples (0.0 if total_samples is 0).
                - overall_comet (float): Average COMET score across language pairs.
                - overall_bleu (float): Average BLEU score across language pairs.
                - overall_chrf (float): Average chrF score across language pairs.
                - per_pair (Dict[str, Dict]): Mapping from language pair to aggregated metrics and sample_count;
                    each value contains `comet`, `bleu`, `chrf`, and `sample_count`.
                - per_language (Dict[str, Dict]): Mapping from language code to aggregated metrics;
                    each value contains `comet`, `bleu`, and `chrf`.
        """
        if not all_metrics:
            return {
                "total_samples": 0,
                "valid_samples": 0,
                "error_count": 0,
                "overall_comet": 0.0,
                "overall_bleu": 0.0,
                "overall_chrf": 0.0,
            }
        
        # CRITICAL: Calculate COMET scores in batch FIRST
        # This fills in the comet=None values from calculate()
        logger.info("Calculating batch COMET scores...")
        metrics_with_comet = self.metrics_calculator.aggregate(all_metrics)
        
        # Use the metrics with COMET scores filled in
        # The aggregate() method returns aggregated scores, but we need per-sample
        # So we use the updated all_metrics (modified in-place by aggregate)
        
        # Basic counts
        total_samples = len(all_metrics)
        valid_samples = sum(1 for m in all_metrics if m.get("valid", False))
        error_count = sum(1 for m in all_metrics if not m.get("valid", False))
        
        # Group by language pair and direction
        by_pair_direction = defaultdict(lambda: defaultdict(list))
        
        for m in all_metrics:
            if not m.get("valid", False):
                continue
            
            pair = m.get("language_pair", "unknown")
            direction = m.get("direction", "unknown")
            by_pair_direction[pair][direction].append(m)
        
        # Aggregate by pair (average both directions)
        per_pair = {}
        for pair, directions in by_pair_direction.items():
            pair_metrics = {}
            
            for metric in ["comet", "bleu", "chrf"]:
                direction_averages = []
                
                for direction, metrics_list in directions.items():
                    # Filter out None values explicitly
                    values = [m.get(metric, 0.0) for m in metrics_list 
                             if metric in m and m.get(metric) is not None]
                    if values:
                        direction_averages.append(sum(values) / len(values))
                
                # Average across directions
                if direction_averages:
                    pair_metrics[metric] = sum(direction_averages) / len(direction_averages)
                else:
                    pair_metrics[metric] = 0.0
            
            pair_metrics["sample_count"] = sum(len(metrics) for metrics in directions.values())
            per_pair[pair] = pair_metrics
        
        # Overall average (across all pairs)
        overall_metrics = {}
        for metric in ["comet", "bleu", "chrf"]:
            values = [pair_metrics[metric] for pair_metrics in per_pair.values() 
                     if metric in pair_metrics and pair_metrics[metric] is not None]
            overall_metrics[f"overall_{metric}"] = sum(values) / len(values) if values else 0.0
        
        # Group by language (for leaderboard)
        by_language = defaultdict(list)
        for pair, pair_metrics in per_pair.items():
            # Extract languages from pair
            # Handle both formats: "en_es" and "eng_spa" or "spa_Latn-arb_Arab"
            if "_" in pair or "-" in pair:
                # Split by underscore or dash
                parts = pair.replace("-", "_").split("_")
                
                # Extract base language codes (first 3 chars usually)
                # eng_spa -> eng, spa
                # spa_Latn-arb_Arab -> spa, arb
                lang1 = parts[0][:3] if len(parts[0]) > 2 else parts[0]
                lang2 = parts[-1][:3] if len(parts) > 1 and len(parts[-1]) > 2 else (parts[-1] if len(parts) > 1 else "")
                
                # Add metrics to both languages
                if lang1:
                    by_language[lang1].append(pair_metrics)
                if lang2 and lang2 != lang1:
                    by_language[lang2].append(pair_metrics)
        
        # Average by language
        per_language = {}
        for lang, metrics_list in by_language.items():
            lang_metrics = {}
            for metric in ["comet", "bleu", "chrf"]:
                values = [m[metric] for m in metrics_list 
                         if metric in m and m[metric] is not None]
                lang_metrics[metric] = sum(values) / len(values) if values else 0.0
            per_language[lang] = lang_metrics
        
        return {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "error_count": error_count,
            "error_rate": error_count / total_samples if total_samples > 0 else 0.0,
            **overall_metrics,
            "per_pair": per_pair,
            "per_language": per_language,
        }
