"""Base class for Spanish tasks.

This module provides a base class that wraps Spanish task classes
to conform to the BaseTask protocol.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from .metrics import SpanishMetricsCalculator

logger = logging.getLogger(__name__)


class SpanishTaskBase:
    """Base class for Spanish tasks.
    
    This provides the calculate_metrics and aggregate_metrics methods
    required by the BaseTask protocol, using the SpanishMetricsCalculator.
    
    Subclasses should implement:
    - load()
    - get_samples()
    - get_prompt()
    - get_task_name()
    - task_type property (returns "multiple_choice" or "generate_until")
    """
    
    def __init__(self, config: Dict):
        """Initialize the Spanish task base.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._metrics_calc = None
    
    @property
    def metrics_calculator(self) -> SpanishMetricsCalculator:
        """Lazy initialization of metrics calculator."""
        if self._metrics_calc is None:
            self._metrics_calc = SpanishMetricsCalculator(self.config)
        return self._metrics_calc
    
    @property
    def task_type(self) -> str:
        """Return task type: 'multiple_choice' or 'generate_until'.
        
        Subclasses should override this.
        """
        return "multiple_choice"

    @property
    def answer_type(self) -> str:
        """Return expected answer type for compatibility checks."""
        if self.task_type == "multiple_choice":
            return "multiple_choice"
        return "freeform"

    @property
    def requires_logprobs(self) -> bool:
        """Multiple-choice Spanish tasks require logprobs-based scoring."""
        return self.answer_type == "multiple_choice"
    
    def get_prompt_for_logprobs(self, sample: Dict) -> Tuple[str, str]:
        """Get prompt format optimized for logprobs evaluation.
        
        Uses the regular prompt and ensures the answer instruction expects
        a single letter choice for logprob scoring.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Tuple of (system_prompt, user_prompt) with a letter-only answer cue.
        """
        system_prompt, user_prompt = self.get_prompt(sample)
        
        if "Respuesta:" in user_prompt:
            prefix, _ = user_prompt.rsplit("Respuesta:", 1)
            user_prompt = f"{prefix}Respuesta (solo la letra):"
        elif "Answer:" in user_prompt:
            prefix, _ = user_prompt.rsplit("Answer:", 1)
            user_prompt = f"{prefix}Answer (letter only):"
        else:
            user_prompt = user_prompt.rstrip() + "\n\nRespuesta (solo la letra):"
        
        return system_prompt, user_prompt
    
    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for a single prediction.
        
        Uses the SpanishMetricsCalculator for consistency.
        
        Args:
            prediction: Model output (text response)
            expected: Expected output (choice index or number)
            sample: Full sample dict
            error: Error message if generation failed (optional)
            error_type: Type of error ('connectivity_error' or 'invalid_response') (optional)
            
        Returns:
            Metrics dictionary with acc/exact_match, valid
        """
        return self.metrics_calculator.calculate(
            prediction=prediction,
            expected=expected,
            sample=sample,
            error=error,
            error_type=error_type,
            task_type=self.task_type,
        )
    
    def get_error_metrics(
        self,
        error: str,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get error metrics structure for failed predictions.
        
        Returns the same structure as SpanishMetricsCalculator would for error cases.
        
        Args:
            error: Error message
            error_type: Type of error ('connectivity_error' or 'invalid_response')
            
        Returns:
            Dictionary of error metrics matching Spanish task format
        """
        return self.metrics_calculator.calculate(
            prediction=None,
            expected="",
            sample={},
            error=error,
            error_type=error_type,
            task_type=self.task_type,
        )
    
    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate per-sample metrics into summary statistics.
        
        Args:
            all_metrics: List of per-sample metric dictionaries
            
        Returns:
            Aggregated metrics
        """
        # Use weight_by_size=True for consistency with spanish.yaml aggregate_metric_list
        return self.metrics_calculator.aggregate(all_metrics, weight_by_size=True)
    
    @property
    def is_multimodal(self) -> bool:
        """Spanish tasks are text-only."""
        return False
    
    @property
    def requires_schema(self) -> bool:
        """Spanish tasks don't use JSON schemas."""
        return False
