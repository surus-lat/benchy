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


class TranslationMetricsCalculator:
    """Calculator for translation metrics: BLEU, chrF, and COMET."""
    
    def __init__(self, config: Dict):
        """Initialize the metrics calculator.
        
        Args:
            config: Configuration dictionary (may contain metric settings)
        """
        self.config = config
        self._comet_model = None  # Lazy-loaded COMET model
        self._comet_pending = []  # Accumulate samples for batch COMET calculation
        
        if not SACREBLEU_AVAILABLE:
            logger.warning("sacrebleu not available. BLEU and chrF metrics will fail.")
        if not COMET_AVAILABLE:
            logger.warning("unbabel-comet not available. COMET metric will fail.")
    
    def _get_comet_model_instance(self):
        """Get COMET model instance (lazy-loaded, cached)."""
        if self._comet_model is None and COMET_AVAILABLE:
            from comet import download_model, load_from_checkpoint
            logger.info("Loading COMET model (this happens once)...")
            model_name = "Unbabel/wmt22-comet-da"
            model_path = download_model(model_name)
            self._comet_model = load_from_checkpoint(model_path)
            logger.info("COMET model loaded successfully")
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
            }
        
        metrics = {
            "valid": True,
            "bleu": 0.0,
            "chrf": 0.0,
            "comet": 0.0,
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
                metrics["valid"] = False
                metrics["error"] = f"BLEU/chrF calculation error: {e}"
        else:
            logger.warning("sacrebleu not available, skipping BLEU and chrF")
        
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
            }
        
        valid_metrics = [m for m in all_metrics if m.get("valid", False)]
        total_samples = len(all_metrics)
        valid_samples = len(valid_metrics)
        
        if valid_samples == 0:
            return {
                "total_samples": total_samples,
                "valid_samples": 0,
                "bleu": 0.0,
                "chrf": 0.0,
                "comet": 0.0,
                "error_rate": 1.0,
            }
        
        # Calculate BLEU and chrF (already done per-sample)
        bleu_scores = [float(m.get("bleu", 0.0)) for m in valid_metrics]
        chrf_scores = [float(m.get("chrf", 0.0)) for m in valid_metrics]
        
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
                            except Exception as e:
                                logger.warning(f"Error calculating COMET batch {i//batch_size + 1}: {e}")
                                all_comet_scores.extend([0.0] * len(batch))
                        
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
                else:
                    logger.debug("No valid COMET data to process")
                    comet_scores = [0.0] * len(valid_metrics)
            except Exception as e:
                logger.warning(f"Error in batch COMET calculation: {e}")
                comet_scores = [0.0] * len(valid_metrics)
        else:
            comet_scores = [0.0] * len(valid_metrics)
        
        aggregated = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "error_rate": (total_samples - valid_samples) / total_samples if total_samples > 0 else 0.0,
            "bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
            "chrf": sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0.0,
            "comet": sum(comet_scores) / len(comet_scores) if comet_scores else 0.0,
        }
        
        return aggregated

