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

**Rubric fields** are the one metric that is not a function of `(predicted,
expected)` alone. A rubric is analytic: named criteria, each with a weight and its
own graded levels (a holistic rubric is the one-criterion case). A grader AI-system
reads the answer against the rubric and the row's answer key and returns an integer
grade per criterion (docs/engine-v1/RUBRIC-QA.md). Grading is I/O, so `run` does it;
this module fixes what a grader is asked (`grader_program`, `grader_input`) and how
its grades become one field score, which keeps this module pure:

    c_ij = Σ_k w_k r_k / Σ_k w_k        r_k = (g_k - g_k,min) / (g_k,max - g_k,min)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from benchy import metrics, types

__all__ = [
    "score_example", "benchmark_score", "value_at",
    "RUBRIC", "GRADE", "grader_program", "grader_input", "rubric_score",
]

#: The graded metric's name. Deliberately not in `metrics.METRICS`: that registry is
#: pure `(predicted, expected) -> float`, and is vendored byte-identical elsewhere.
RUBRIC = "rubric"

#: What a grader returns for each criterion, as an IR node. The justification comes
#: first: a model writes structured output in schema order, so it reasons before it
#: commits to a grade instead of rationalizing one it already gave.
GRADE = {"type": "object", "fields": {"justification": {"type": "string"}, "grade": {"type": "int"}}}


def score_example(
    prediction: Mapping, expected: Mapping, ir: Mapping, grades: Mapping | None = None,
) -> tuple[list[dict], float]:
    """Return `(field_scores, instance_score)` for one valid prediction.

    `grades` maps a rubric dimension's path (as a tuple) to its grader's validated
    output, one `GRADE` per criterion; those are kept beside the field score.

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
        if name == RUBRIC:
            graded = grades[tuple(dimension["path"])]
            value = rubric_score(graded, dimension["params"]["criteria"])
            field_scores.append({"path": dimension["path"], "score": value, "weight": dimension["weight"], "criteria": graded})
            continue
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


def grader_program(ir: Mapping) -> dict:
    """The program the benchmark's grader implements: an ordinary IR program.

    It sees the question exactly as the AI-system did, the row's answer key for the
    field, the answer, and the rubric rendered as text; it returns one `GRADE` per
    criterion, named as the rubric names them.
    """
    text = {"type": "string"}
    (dimension,) = (d for d in ir["scoring"]["dimensions"] if d.get("metric") == RUBRIC)
    criteria = dimension["params"]["criteria"]
    return {
        "input": {"type": "object", "fields": {"input": ir["program"]["input"], "reference": text, "answer": text, "rubric": text}},
        "output": {"type": "object", "fields": {criterion["name"]: GRADE for criterion in criteria}},
    }


def grader_input(inputs: Mapping, reference: object, answer: object, criteria: Sequence[Mapping]) -> dict:
    blocks = []
    for criterion in criteria:
        head = f"{criterion['name']}: {criterion['description']}" if criterion.get("description") else criterion["name"]
        blocks.append("\n".join([head] + [f"  {level['grade']}: {level['description']}" for level in criterion["levels"]]))
    return {"input": inputs, "reference": reference, "answer": answer, "rubric": "\n\n".join(blocks)}


def rubric_score(grades: Mapping, criteria: Sequence[Mapping]) -> float:
    """Weighted mean of the criteria, each grade mapped linearly from its lowest level (0) to its highest (1)."""
    total = sum(c["weight"] for c in criteria)
    return sum(c["weight"] * _level_score(grades[c["name"]]["grade"], c["levels"]) for c in criteria) / total


def _level_score(grade: int, levels: Sequence[Mapping]) -> float:
    low, high = levels[0]["grade"], levels[-1]["grade"]
    return (grade - low) / (high - low)

