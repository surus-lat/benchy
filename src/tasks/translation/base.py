"""Base class for translation tasks.

This module provides a base class that wraps translation task classes
to conform to the BaseTask protocol.
"""

import logging
from typing import Dict, Any, List, Optional

from .metrics import TranslationMetricsCalculator

logger = logging.getLogger(__name__)


class TranslationTaskBase:
    """Base class for translation tasks.
    
    This provides the calculate_metrics and aggregate_metrics methods
    required by the BaseTask protocol, using the TranslationMetricsCalculator.
    
    Subclasses should implement:
    - load()
    - get_samples()
    - get_prompt()
    - get_task_name()
    """
    
    def __init__(self, config: Dict):
        """Initialize the translation task base.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._metrics_calc = None
    
    @property
    def metrics_calculator(self) -> TranslationMetricsCalculator:
        """Lazy initialization of metrics calculator."""
        if self._metrics_calc is None:
            self._metrics_calc = TranslationMetricsCalculator(self.config)
        return self._metrics_calc
    
    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for a single prediction.
        
        Uses the TranslationMetricsCalculator for consistency.
        Simply passes error/error_type through to the calculator.
        
        Args:
            prediction: Model output (translation text)
            expected: Expected output (reference translation)
            sample: Full sample dict
            error: Error message if generation failed (optional)
            error_type: Type of error ('connectivity_error' or 'invalid_response') (optional)
            
        Returns:
            Metrics dictionary with bleu, chrf, comet, valid
        """
        # Pass through to metrics calculator - it handles error cases
        return self.metrics_calculator.calculate(
            prediction=prediction,
            expected=expected,
            error=error,
            error_type=error_type,
        )
    
    def get_error_metrics(
        self,
        error: str,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get error metrics structure for failed predictions.
        
        Returns the same structure as TranslationMetricsCalculator would for error cases,
        ensuring consistency with calculate_metrics() output.
        
        Args:
            error: Error message
            error_type: Type of error ('connectivity_error' or 'invalid_response')
            
        Returns:
            Dictionary of error metrics matching translation format
        """
        # Use metrics calculator to get consistent error structure
        return self.metrics_calculator.calculate(
            prediction=None,
            expected="",
            error=error,
            error_type=error_type,
        )
    
    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate per-sample metrics into summary statistics.
        
        Args:
            all_metrics: List of per-sample metric dictionaries
            
        Returns:
            Aggregated metrics
        """
        return self.metrics_calculator.aggregate(all_metrics)
    
    @property
    def is_multimodal(self) -> bool:
        """Translation tasks are text-only."""
        return False
    
    @property
    def requires_schema(self) -> bool:
        """Translation tasks don't use JSON schemas."""
        return False

