"""BaseScorer — the ABC every scorer in this package extends.

A scorer is a small, composable node in a symbolic expression tree (see the
package docstring in `benchy/scoring/__init__.py`). `BaseScorer` exists so a
new primitive costs ~10 lines: define `name`, implement `evaluate`, and
override `__repr__` so `parse_scorer(repr(scorer))` can rebuild it. `fitness`
and `aggregate` come for free.

Convention enforced everywhere in this package: `Score.value` and
`fitness(...)` are always a float in `[0.0, 1.0]` where **higher is better**,
even for metrics that are naturally "lower is better" in the literature
(word error rate, character error rate, mean squared error). Those scorers
invert internally — see their docstrings for the exact formula — so that a
generic optimizer consuming `fitness` never has to know which primitives are
error-family and which aren't. Getting this backwards silently breaks every
downstream optimizer, so it is called out loudly on every scorer that does it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from benchy.core import Sample, Score

__all__ = ["BaseScorer"]


def _mean(values: Sequence[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def clamp01(value: float) -> float:
    """Clamp a float into [0.0, 1.0]. Shared by every primitive that needs it."""
    if value != value:  # NaN check without importing math
        return 0.0
    return max(0.0, min(1.0, value))


class BaseScorer(ABC):
    """The ABC every scorer extends.

    Subclasses must:
      - set a class or instance attribute `name: str`.
      - implement `evaluate(prediction, expected, sample) -> Score`.
      - override `__repr__` to emit an expression `parse_scorer` can rebuild
        (see `benchy.scoring.registry`). The default `__repr__` here raises
        `NotImplementedError` on purpose, so a scorer that forgets to
        implement round-tripping fails loudly the first time anyone prints
        or persists it, instead of silently breaking `parse_scorer` later.

    `fitness` and `aggregate` are given sane defaults so most primitives
    never need to override them.
    """

    name: str

    @abstractmethod
    def evaluate(self, prediction: Any, expected: Any, sample: Sample | None = None) -> Score:
        """Grade one (prediction, expected) pair. Rich; carries a `breakdown`."""
        raise NotImplementedError

    def fitness(self, prediction: Any, expected: Any, sample: Sample | None = None) -> float:
        """The single scalar an optimizer consumes. Defaults to `evaluate(...).value`."""
        return self.evaluate(prediction, expected, sample).value

    def aggregate(self, scores: Sequence[Score]) -> Mapping[str, Any]:
        """Summarize a run. Default: mean of `.value`, under the `"fitness"` key."""
        return {"fitness": _mean([s.value for s in scores]), "n": len(scores)}

    def __repr__(self) -> str:  # pragma: no cover - exercised via subclass override
        raise NotImplementedError(
            f"{type(self).__name__} must override __repr__ with a round-trippable "
            "expression so parse_scorer(repr(scorer)) can reconstruct it."
        )
