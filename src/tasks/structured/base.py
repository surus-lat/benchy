"""Base class for structured extraction tasks.

This module provides a base class that wraps existing task classes
(ParaloqTask, ChatExtractTask) to conform to the BaseTask protocol.
"""

from typing import Dict, Any, List, Optional
from .metrics import MetricsCalculator


class StructuredExtractionTaskBase:
    """Mixin that adds metrics calculation to structured extraction tasks.
    
    This provides the calculate_metrics and aggregate_metrics methods
    required by the BaseTask protocol, using the existing MetricsCalculator.
    
    Subclasses should implement:
    - load()
    - get_samples()
    - get_prompt()
    - get_task_name()
    """
    
    def __init__(self, config: Dict):
        """Initialize the task base.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._metrics_calc = None
    
    @property
    def metrics_calculator(self) -> MetricsCalculator:
        """Lazy initialization of metrics calculator."""
        if self._metrics_calc is None:
            self._metrics_calc = MetricsCalculator(self.config)
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
        
        Uses the existing MetricsCalculator for consistency.
        Simply passes error/error_type through to the calculator.
        
        Args:
            prediction: Model output
            expected: Expected output
            sample: Full sample dict (contains schema)
            error: Error message if generation failed (optional)
            error_type: Type of error ('connectivity_error' or 'invalid_response') (optional)
            
        Returns:
            Metrics dictionary
        """
        # Pass through to metrics calculator - it handles error cases
        return self.metrics_calculator.calculate_all(
            prediction=prediction,
            expected=expected,
            schema=sample.get("schema", {}),
            error=error,
            error_type=error_type,
        )
    
    def get_error_metrics(
        self,
        error: str,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get error metrics structure for failed predictions.
        
        Returns the same structure as MetricsCalculator would for error cases,
        ensuring consistency with calculate_metrics() output.
        
        Args:
            error: Error message
            error_type: Type of error ('connectivity_error' or 'invalid_response')
            
        Returns:
            Dictionary of error metrics matching structured extraction format
        """
        # Use metrics calculator to get consistent error structure
        return self.metrics_calculator.calculate_all(
            prediction=None,
            expected={},
            schema={},
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
        return self.metrics_calculator.aggregate_metrics(all_metrics)
    
    @property
    def is_multimodal(self) -> bool:
        """Structured extraction tasks are text-only."""
        return False
    
    @property
    def requires_schema(self) -> bool:
        """Structured extraction tasks use JSON schemas."""
        return True

    @property
    def answer_type(self) -> str:
        """Structured extraction tasks return JSON outputs."""
        return "structured"

    @property
    def requires_logprobs(self) -> bool:
        """Structured extraction tasks do not require logprobs."""
        return False
