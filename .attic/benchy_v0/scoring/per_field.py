"""Per-field scoring — bare metal."""

from __future__ import annotations

from typing import Any


def per_field_scorer(case_sensitive: bool = False, numeric_tolerance: float = 0.0):
    """Build a per-field scorer."""

    def score(expected: Any, actual: Any) -> float:
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return 0.0

        if not expected:
            return 1.0 if not actual else 0.0

        field_scores: list[float] = []
        for key, exp_val in expected.items():
            act_val = actual.get(key)
            field_scores.append(_score_field(exp_val, act_val))

        return sum(field_scores) / len(field_scores) if field_scores else 0.0

    def _score_field(expected: Any, actual: Any) -> float:
        if expected is None and actual is None:
            return 1.0
        if expected is None or actual is None:
            return 0.0

        if isinstance(expected, str) and isinstance(actual, str):
            if not case_sensitive:
                return 1.0 if expected.lower() == actual.lower() else 0.0
            return 1.0 if expected == actual else 0.0

        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if numeric_tolerance > 0:
                return 1.0 if abs(expected - actual) <= numeric_tolerance else 0.0
            return 1.0 if expected == actual else 0.0

        if isinstance(expected, dict) and isinstance(actual, dict):
            return score(expected, actual)

        if isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                return 0.0
            if not expected:
                return 1.0
            return sum(_score_field(e, a) for e, a in zip(expected, actual)) / len(expected)

        return 1.0 if expected == actual else 0.0

    def aggregate(scores: list[float]) -> dict[str, float]:
        if not scores:
            return {"score": 0.0, "mean_field_score": 0.0, "count": 0}
        return {
            "score": sum(scores) / len(scores),
            "mean_field_score": sum(scores) / len(scores),
            "count": len(scores),
        }

    return score, aggregate
