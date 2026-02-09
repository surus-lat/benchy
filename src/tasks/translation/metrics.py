"""Translation metrics calculator.

Calculates BLEU, chrF, and COMET scores for translation evaluation.
"""

import logging
from typing import Dict, Any, List, Optional

try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
    # Global COMET model cache
    _COMET_MODEL = None
    _COMET_MODEL_PATH = None
except ImportError:
    COMET_AVAILABLE = False
    _COMET_MODEL = None
    _COMET_MODEL_PATH = None

logger = logging.getLogger(__name__)


def _get_comet_model():
    """Get or initialize the global COMET model."""
    global _COMET_MODEL, _COMET_MODEL_PATH
    
    if not COMET_AVAILABLE:
        raise ImportError("COMET is not available. Please install it with: pip install unbabel-comet")
    
    model_name = "Unbabel/wmt22-comet-da"
    
    if _COMET_MODEL is None or _COMET_MODEL_PATH != model_name:
        logger.info(f"Loading COMET model: {model_name}")
        _COMET_MODEL_PATH = download_model(model_name)
        _COMET_MODEL = load_from_checkpoint(_COMET_MODEL_PATH)
        _COMET_MODEL_PATH = model_name
        logger.info("COMET model loaded successfully")
    
    return _COMET_MODEL


def load_comet_model():
    """Load the global COMET model once and return it."""
    if not COMET_AVAILABLE:
        logger.warning("unbabel-comet not available. COMET metric will be skipped.")
        return None
    return _get_comet_model()


class TranslationMetricsCalculator:
    """Calculator for translation metrics: BLEU, chrF, and COMET."""
    
    def __init__(self, config: Dict):
        """Initialize the metrics calculator.
        
        Args:
            config: Configuration dictionary (may contain metric settings)
        """
        self.config = config
        self._comet_model = config.get("comet_model")  # Optional preloaded model
        self._comet_pending = []  # Accumulate samples for batch COMET calculation
        
        if not SACREBLEU_AVAILABLE:
            logger.warning("sacrebleu not available. BLEU and chrF metrics will fail.")
        if not COMET_AVAILABLE:
            logger.warning("unbabel-comet not available. COMET metric will fail.")
    
    def _get_comet_model_instance(self):
        """Get COMET model instance (lazy-loaded, cached)."""
        if self._comet_model is None and COMET_AVAILABLE:
            self._comet_model = load_comet_model()
        return self._comet_model
    
    def calculate(
        self,
        prediction: Optional[str],
        expected: str,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for a single translation pair.
        
        Args:
            prediction: Model translation output (or None on error)
            expected: Reference translation
            error: Error message if generation failed (optional)
            error_type: Type of error ('connectivity_error' or 'invalid_response') (optional)
            
        Returns:
            Dictionary with metrics: {"bleu": float, "chrf": float, "comet": float, "valid": bool}
        """
        # Handle errors
        if error or prediction is None:
            return {
                "valid": False,
                "error": error or "No prediction",
                "error_type": error_type,
                "bleu": 0.0,
                "chrf": 0.0,
                "comet": 0.0,
                "metric_degraded": False,
                "metric_warnings": [],
            }
        
        # Normalize strings
        pred_text = str(prediction).strip() if prediction else ""
        ref_text = str(expected).strip() if expected else ""
        
        if not pred_text or not ref_text:
            return {
                "valid": False,
                "error": "Empty prediction or reference",
                "error_type": "invalid_response",
                "bleu": 0.0,
                "chrf": 0.0,
                "comet": 0.0,
                "metric_degraded": False,
                "metric_warnings": [],
            }
        
        metrics = {
            "valid": True,
            "bleu": 0.0,
            "chrf": 0.0,
            "comet": 0.0,
            "metric_degraded": False,
            "metric_warnings": [],
            "_text_pred": pred_text,
            "_text_ref": ref_text,
        }
        
        # Calculate BLEU and chrF (sacrebleu)
        if SACREBLEU_AVAILABLE:
            try:
                # BLEU
                bleu_score = sacrebleu.sentence_bleu(pred_text, [ref_text])
                metrics["bleu"] = bleu_score.score / 100.0  # Normalize to 0-1
                
                # chrF
                chrf_score = sacrebleu.sentence_chrf(pred_text, [ref_text])
                metrics["chrf"] = chrf_score.score / 100.0  # Normalize to 0-1
            except Exception as e:
                logger.warning(f"Error calculating BLEU/chrF: {e}")
                metrics["metric_degraded"] = True
                metrics["metric_warnings"].append(f"bleu_chrf_calculation_error: {e}")
        else:
            logger.warning("sacrebleu not available, skipping BLEU and chrF")
            metrics["metric_degraded"] = True
            metrics["metric_warnings"].append("sacrebleu_not_available")
        
        # Defer COMET calculation - we'll batch it later
        # Store prediction/ref for batch processing in aggregate_metrics
        if COMET_AVAILABLE:
            # Store for batch calculation (we'll calculate in aggregate_metrics)
            metrics["_comet_pred"] = pred_text
            metrics["_comet_ref"] = ref_text
            metrics["comet"] = None  # Will be filled in aggregate_metrics
        else:
            logger.debug("COMET not available, skipping")
            metrics["comet"] = 0.0
            metrics["metric_degraded"] = True
            metrics["metric_warnings"].append("comet_not_available")
        
        return metrics
    
    def aggregate(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate per-sample metrics into corpus-level statistics.
        
        For translation, we use corpus-level BLEU/chrF (sacrebleu style)
        and average COMET scores. COMET is calculated in batches here.
        
        Args:
            all_metrics: List of per-sample metric dictionaries
            
        Returns:
            Aggregated metrics dictionary
        """
        if not all_metrics:
            return {
                "total_samples": 0,
                "valid_samples": 0,
                "bleu": 0.0,
                "chrf": 0.0,
                "comet": 0.0,
                "metric_degraded": False,
                "metric_warnings": [],
                "bleu_aggregation": "corpus",
                "chrf_aggregation": "corpus",
            }

        valid_metrics = [m for m in all_metrics if m.get("valid", False)]
        total_samples = len(all_metrics)
        valid_samples = len(valid_metrics)
        metric_warnings = []
        for m in all_metrics:
            for warning in m.get("metric_warnings", []):
                if warning:
                    metric_warnings.append(str(warning))

        if valid_samples == 0:
            return {
                "total_samples": total_samples,
                "valid_samples": 0,
                "bleu": 0.0,
                "chrf": 0.0,
                "comet": 0.0,
                "error_rate": 1.0,
                "metric_degraded": bool(metric_warnings),
                "metric_warnings": sorted(set(metric_warnings)),
                "bleu_aggregation": "corpus",
                "chrf_aggregation": "corpus",
            }

        # Calculate corpus-level BLEU and chrF.
        bleu_value = 0.0
        chrf_value = 0.0
        if SACREBLEU_AVAILABLE:
            try:
                pred_texts = []
                ref_texts = []
                for m in valid_metrics:
                    pred = m.get("_text_pred")
                    ref = m.get("_text_ref")
                    if pred and ref:
                        pred_texts.append(str(pred))
                        ref_texts.append(str(ref))

                if pred_texts and len(pred_texts) == len(ref_texts):
                    bleu_value = float(sacrebleu.corpus_bleu(pred_texts, [ref_texts]).score) / 100.0
                    chrf_value = float(sacrebleu.corpus_chrf(pred_texts, [ref_texts]).score) / 100.0
                else:
                    metric_warnings.append("missing_text_for_corpus_bleu_chrf")
            except Exception as e:
                logger.warning(f"Error calculating corpus BLEU/chrF: {e}")
                metric_warnings.append(f"corpus_bleu_chrf_calculation_error: {e}")
        else:
            metric_warnings.append("sacrebleu_not_available")

        # Calculate COMET in batches (much faster than per-sample)
        comet_scores = []
        if COMET_AVAILABLE:
            try:
                # Collect all predictions and references for batch processing
                comet_data = []
                for m in valid_metrics:
                    pred = m.get("_comet_pred")
                    ref = m.get("_comet_ref")
                    if pred and ref:
                        comet_data.append({"src": "", "mt": pred, "ref": ref})
                
                if comet_data:
                    logger.info(f"Calculating COMET scores for {len(comet_data)} samples in batch...")
                    model = self._get_comet_model_instance()
                    if model:
                        # Process in batches of 32 for efficiency
                        batch_size = 32
                        all_comet_scores = []
                        
                        for i in range(0, len(comet_data), batch_size):
                            batch = comet_data[i:i + batch_size]
                            try:
                                comet_output = model.predict(
                                    batch,
                                    batch_size=len(batch),
                                    gpus=0,  # CPU-only for now
                                )
                                
                                if hasattr(comet_output, 'scores') and comet_output.scores:
                                    all_comet_scores.extend(comet_output.scores)
                                elif hasattr(comet_output, 'system_score'):
                                    # If only system_score, distribute it (shouldn't happen with batch)
                                    all_comet_scores.extend([comet_output.system_score] * len(batch))
                                else:
                                    logger.warning(f"Unexpected COMET output format: {type(comet_output)}")
                                    all_comet_scores.extend([0.0] * len(batch))
                                    metric_warnings.append("unexpected_comet_output_format")
                            except Exception as e:
                                logger.warning(f"Error calculating COMET batch {i//batch_size + 1}: {e}")
                                all_comet_scores.extend([0.0] * len(batch))
                                metric_warnings.append(f"comet_batch_calculation_error: {e}")
                        
                        # Map scores back to metrics
                        comet_scores = [float(s) if isinstance(s, (int, float)) else 0.0 for s in all_comet_scores]
                        
                        # Update metrics with COMET scores
                        comet_idx = 0
                        for m in valid_metrics:
                            if m.get("_comet_pred") and m.get("_comet_ref"):
                                if comet_idx < len(comet_scores):
                                    m["comet"] = comet_scores[comet_idx]
                                    comet_idx += 1
                                else:
                                    m["comet"] = 0.0
                    else:
                        logger.warning("COMET model not available, skipping COMET calculation")
                        comet_scores = [0.0] * len(valid_metrics)
                        metric_warnings.append("comet_model_not_available")
                else:
                    logger.debug("No valid COMET data to process")
                    comet_scores = [0.0] * len(valid_metrics)
                    metric_warnings.append("missing_text_for_comet")
            except Exception as e:
                logger.warning(f"Error in batch COMET calculation: {e}")
                comet_scores = [0.0] * len(valid_metrics)
                metric_warnings.append(f"comet_aggregate_error: {e}")
        else:
            comet_scores = [0.0] * len(valid_metrics)
            metric_warnings.append("comet_not_available")

        deduped_warnings = sorted(set(metric_warnings))

        aggregated = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "error_rate": (total_samples - valid_samples) / total_samples if total_samples > 0 else 0.0,
            "bleu": bleu_value,
            "chrf": chrf_value,
            "comet": sum(comet_scores) / len(comet_scores) if comet_scores else 0.0,
            "metric_degraded": bool(deduped_warnings),
            "metric_warnings": deduped_warnings,
            "bleu_aggregation": "corpus",
            "chrf_aggregation": "corpus",
        }

        return aggregated
