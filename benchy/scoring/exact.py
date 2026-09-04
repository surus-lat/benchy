"""Exact match scoring — bare metal."""

from __future__ import annotations

from typing import Any


def exact_match_scorer(case_sensitive: bool = True):
    """Build an exact-match scorer."""

    def score(expected: Any, actual: Any) -> float:
        if expected is None and actual is None:
            return 1.0
        if expected is None or actual is None:
            return 0.0

        if isinstance(expected, str) and isinstance(actual, str):
            if not case_sensitive:
                return 1.0 if expected.lower() == actual.lower() else 0.0
            return 1.0 if expected == actual else 0.0

        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return 1.0 if expected == actual else 0.0

        if isinstance(expected, dict) and isinstance(actual, dict):
            if expected.keys() != actual.keys():
                return 0.0
            return 1.0 if all(score(expected[k], actual[k]) == 1.0 for k in expected) else 0.0

        if isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                return 0.0
            return 1.0 if all(score(e, a) == 1.0 for e, a in zip(expected, actual)) else 0.0

        return 1.0 if expected == actual else 0.0

    def aggregate(scores: list[float]) -> dict[str, float]:
        if not scores:
            return {"score": 0.0, "accuracy": 0.0, "count": 0}
        return {
            "score": sum(scores) / len(scores),
            "accuracy": sum(1 for s in scores if s == 1.0) / len(scores),
            "count": len(scores),
        }

    return score, aggregate
