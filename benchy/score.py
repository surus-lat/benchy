"""Scoring: field correctness, instance score, benchmark score (paper §4, §8).

Pure arithmetic over already-validated values — no I/O, no filesystem, nothing to
mock. The three levels of the paper map to the three functions here.

Field correctness is an indicator under the canonical metric:

    c_ij = 1[ŷ_ij == y*_ij]

or a float in [0, 1] under a metric declared in `scoring.field_metrics` (the closed
registry in `benchy.metrics`). The aggregation below does not change either way:

Instance score is the normalized weighted mean over scoring dimensions:

    s_i = Σ_j w_j c_ij / Σ_j w_j

Benchmark score is the arithmetic mean of *contributions*, where a failed example
contributes zero but stays in the denominator:

    B = (1/N) Σ_i q_i        q_i = s_i if valid else 0

The distinction between a stored `null` score and a zero contribution is the whole
point of paper A.13: it keeps "the system answered, and was wrong" separable from
"the system did not answer", while refusing to let failures vanish from the average.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from benchy import metrics, types

__all__ = ["score_example", "benchmark_score", "value_at"]


def score_example(prediction: Mapping, expected: Mapping, ir: Mapping) -> tuple[list[dict], float]:
    """Return `(field_scores, instance_score)` for one valid prediction.

    Dimensions are walked in IR order, which is output-schema order, so
    `field_scores` is deterministic across runs and machines. A dimension without a
    declared metric scores with the canonical exact match (paper A.7) — a benchmark
    without `scoring.field_metrics` compiles to exactly those dimensions, so its
    score is identical to one compiled before field metrics existed.
    """
    output_schema = ir["program"]["output"]
    field_scores = []
    for dimension in ir["scoring"]["dimensions"]:
        name = dimension.get("metric", "exact")
        predicted = value_at(prediction, dimension["path"])
        expected_value = value_at(expected, dimension["path"])
        if name == "exact":
            value = 1.0 if types.equal(predicted, expected_value, types.at(output_schema, dimension["path"])) else 0.0
        else:
            value = metrics.score(name, predicted, expected_value, dimension.get("params"))
        field_scores.append({"path": dimension["path"], "score": value, "weight": dimension["weight"]})
    # The compiler guarantees Σw > 0 (spec §6), so this cannot divide by zero.
    total = sum(f["weight"] for f in field_scores)
    return field_scores, sum(f["weight"] * f["score"] for f in field_scores) / total


def benchmark_score(contributions: Sequence[float]) -> float:
    """Arithmetic mean over every example, failures included as zero."""
    # An empty dataset aborts the run (spec §7), so N > 0 here.
    return sum(contributions) / len(contributions)


def value_at(value: Mapping, path: Sequence[str]) -> object:
    for key in path:
        value = value[key]
    return value
