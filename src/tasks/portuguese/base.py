"""Base class for Portuguese tasks.

Provides metric handling and capability flags for Benchy tasks.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from .metrics import PortugueseMetricsCalculator

logger = logging.getLogger(__name__)


class PortugueseTaskBase:
    """Base class for Portuguese tasks.

    Subclasses should implement:
    - load()
    - get_samples()
    - get_prompt()
    - get_task_name()
    - task_type property ("multiple_choice", "classification", or "regression")
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._metrics_calc: Optional[PortugueseMetricsCalculator] = None

    @property
    def metrics_calculator(self) -> PortugueseMetricsCalculator:
        """Lazy init for metrics calculator."""
        if self._metrics_calc is None:
            self._metrics_calc = PortugueseMetricsCalculator(self.config)
        return self._metrics_calc

    @property
    def task_type(self) -> str:
        """Return task type.

        Subclasses should override this.
        """
        return "multiple_choice"

    @property
    def labels(self) -> List[str]:
        """Optional label set for classification tasks."""
        labels = self.config.get("labels")
        if isinstance(labels, list):
            return labels
        return []

    @property
    def score_range(self) -> Optional[Tuple[float, float]]:
        """Optional score range for regression tasks."""
        return self.config.get("score_range")

    @property
    def answer_type(self) -> str:
        """Return expected answer type for compatibility checks."""
        if self.task_type == "multiple_choice":
            return "multiple_choice"
        return "freeform"

    @property
    def requires_logprobs(self) -> bool:
        """Multiple-choice Portuguese tasks require logprobs scoring."""
        return self.answer_type == "multiple_choice"

    def get_prompt_for_logprobs(self, sample: Dict) -> Tuple[str, str]:
        """Return a prompt variant optimized for logprobs scoring."""
        system_prompt, user_prompt = self.get_prompt(sample)

        if "Resposta correta:" in user_prompt:
            prefix, _ = user_prompt.rsplit("Resposta correta:", 1)
            user_prompt = f"{prefix}Resposta correta (apenas a letra):"
        elif "Resposta:" in user_prompt:
            prefix, _ = user_prompt.rsplit("Resposta:", 1)
            user_prompt = f"{prefix}Resposta (apenas a letra):"
        elif "Answer:" in user_prompt:
            prefix, _ = user_prompt.rsplit("Answer:", 1)
            user_prompt = f"{prefix}Answer (letter only):"
        else:
            user_prompt = user_prompt.rstrip() + "\n\nResposta (apenas a letra):"

        return system_prompt, user_prompt

    def calculate_metrics(
        self,
        prediction: Any,
        expected: Any,
        sample: Dict,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate metrics for a single prediction."""
        return self.metrics_calculator.calculate(
            prediction=prediction,
            expected=expected,
            sample=sample,
            error=error,
            error_type=error_type,
            task_type=self.task_type,
            labels=self.labels,
            score_range=self.score_range,
        )

    def get_error_metrics(self, error: str, error_type: Optional[str] = None) -> Dict[str, Any]:
        """Return error metrics structure."""
        return self.metrics_calculator.calculate(
            prediction=None,
            expected="",
            sample={},
            error=error,
            error_type=error_type,
            task_type=self.task_type,
            labels=self.labels,
            score_range=self.score_range,
        )

    def aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Aggregate per-sample metrics."""
        return self.metrics_calculator.aggregate(
            all_metrics,
            task_type=self.task_type,
            labels=self.labels,
        )

    @property
    def requires_multimodal(self) -> bool:
        return False

    @property
    def requires_schema(self) -> bool:
        return False
