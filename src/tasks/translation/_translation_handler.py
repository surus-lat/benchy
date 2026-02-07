"""Translation handler base class.

Extends FreeformHandler with translation-specific features:
- COMET model management (passed via config)
- Bidirectional language pair aggregation
- Translation metrics (BLEU, chrF, COMET)
"""

import logging
import re
from collections import defaultdict
from typing import Dict, Any, List, Optional

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
        """Initialize translation handler.
        
        Args:
            config: Configuration dict, should include:
                - comet_model: Preloaded COMET model (from TaskGroupSpec setup)
                - language_pairs: List of language pairs to evaluate
        """
        super().__init__(config)
        
        # COMET model (passed from setup)
        self.comet_model = self.config.get("comet_model") if self.config else None
        
        # Initialize metrics calculator with COMET model
        self._metrics_calc = None
    
    @property
    def metrics_calculator(self) -> TranslationMetricsCalculator:
        """Lazy initialization of metrics calculator."""
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
        """Calculate translation metrics for a single sample.
        
        Uses TranslationMetricsCalculator which handles:
        - BLEU score
        - chrF score
        - COMET score (if model available)
        - Error tracking
        
        Args:
            prediction: Model translation output
            expected: Reference translation
            sample: Full sample dict (includes language info)
            error: Error message if generation failed
            error_type: Type of error
            
        Returns:
            Metrics dict with bleu, chrf, comet, valid, etc.
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
        """Get error metrics for failed predictions.
        
        Args:
            error: Error message
            error_type: Type of error
            
        Returns:
            Error metrics dict
        """
        return self.metrics_calculator.calculate(
            prediction=None,
            expected="",
            error=error,
            error_type=error_type,
        )
    
    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate translation metrics across samples.
        
        This performs multi-level aggregation:
        1. FIRST: Calculate batch COMET scores (via TranslationMetricsCalculator)
        2. Group by language pair and direction
        3. Average each direction separately
        4. Average both directions for pair score
        5. Average across pairs for overall score
        6. Group by language for per-language scores
        
        Returns:
            Dict with:
            - overall_comet, overall_bleu, overall_chrf
            - per_pair: {pair: {comet, bleu, chrf}}
            - per_language: {lang: {comet, bleu, chrf}}
            - total_samples, valid_samples, error_count
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
        self.metrics_calculator.aggregate(all_metrics)
        
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
        language_code_re = re.compile(r"^[a-z]{2,3}$")
        for pair, pair_metrics in per_pair.items():
            # Extract languages from pair
            # Handle both formats: "en_es" and "eng_spa" or "spa_Latn-arb_Arab"
            if "_" in pair or "-" in pair:
                # Split by underscore or dash
                parts = pair.replace("-", "_").split("_")
                
                # Extract base language codes (first 3 chars usually)
                # eng_spa -> eng, spa
                # spa_Latn-arb_Arab -> spa, arb
                lang1_raw = parts[0]
                lang1 = lang1_raw[:3] if len(lang1_raw) > 2 else lang1_raw
                lang2 = ""
                for part in parts[1:]:
                    if language_code_re.match(part):
                        lang2 = part[:3] if len(part) > 2 else part
                        break
                
                # Add metrics to both languages
                if pair_metrics and lang1:
                    by_language[lang1].append(pair_metrics)
                if pair_metrics and lang2 and lang2 != lang1:
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
