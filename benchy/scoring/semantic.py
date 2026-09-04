"""Semantic similarity scoring — bare metal."""

from __future__ import annotations

from typing import Any


def semantic_scorer(threshold: float = 0.8):
    """Build a semantic scorer."""

    def score(expected: Any, actual: Any) -> float:
        if expected is None or actual is None:
            return 0.0

        exp_str = str(expected).strip().lower()
        act_str = str(actual).strip().lower()

        if not exp_str or not act_str:
            return 0.0

        if exp_str == act_str:
            return 1.0

        if exp_str in act_str or act_str in exp_str:
            return 0.9

        exp_words = set(exp_str.split())
        act_words = set(act_str.split())
        if not exp_words or not act_words:
            return 0.0

        intersection = exp_words & act_words
        union = exp_words | act_words
        jaccard = len(intersection) / len(union) if union else 0.0

        return jaccard if jaccard >= threshold else jaccard * 0.5

    def aggregate(scores: list[float]) -> dict[str, float]:
        if not scores:
            return {"score": 0.0, "mean_similarity": 0.0, "count": 0}
        return {
            "score": sum(scores) / len(scores),
            "mean_similarity": sum(scores) / len(scores),
            "count": len(scores),
        }

    return score, aggregate
