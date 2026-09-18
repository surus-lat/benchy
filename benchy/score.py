"""Scoring: field correctness, instance score, benchmark score (paper §4, §8).

Pure arithmetic over already-validated values — no I/O, no filesystem, nothing to
mock. The three levels of the paper map to the three functions here.

Field correctness is an indicator:

    c_ij = 1[ŷ_ij == y*_ij]

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

from benchy import types

__all__ = ["score_example", "benchmark_score"]


def score_example(prediction: Mapping, expected: Mapping, ir: Mapping) -> tuple[list[dict], float]:
    """Return `(field_scores, instance_score)` for one valid prediction.

    Dimensions are walked in IR order, which is output-schema order, so
    `field_scores` is deterministic across runs and machines.
    """
    output_schema = ir["program"]["output"]
    field_scores = [
        {
            "path": dimension["path"],
            "score": int(
                types.equal(
                    _value_at(prediction, dimension["path"]),
                    _value_at(expected, dimension["path"]),
                    types.at(output_schema, dimension["path"]),
                )
            ),
            "weight": dimension["weight"],
        }
        for dimension in ir["scoring"]["dimensions"]
    ]
    # The compiler guarantees Σw > 0 (spec §6), so this cannot divide by zero.
    total = sum(f["weight"] for f in field_scores)
    return field_scores, sum(f["weight"] * f["score"] for f in field_scores) / total


def benchmark_score(contributions: Sequence[float]) -> float:
    """Arithmetic mean over every example, failures included as zero."""
    # An empty dataset aborts the run (spec §7), so N > 0 here.
    return sum(contributions) / len(contributions)


def _value_at(value: Mapping, path: Sequence[str]) -> object:
    for key in path:
        value = value[key]
    return value
