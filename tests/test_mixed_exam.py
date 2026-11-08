"""Mixed exams: optional input fields (`options?`) and per-row metric routing.

One benchmark may carry open-answer rows and multiple-choice rows together: the
input schema declares `options?` (optional), and the answer field's rubric entry
routes rows that carry `options` to casefold_strip. Open rows are judged by the
grader; choice rows are scored by exact (case-insensitive) match and never reach
the grader at all.

Backward compatibility is the contract: a benchmark without `?` or `route`
compiles to byte-identical IR and scores exactly as before (the rest of the
suite pins that); these tests pin the new shapes.
"""

from __future__ import annotations

import json

import pytest
from conftest import edit

from benchy import types
from benchy.compiler import compile_benchmark, compile_scoring
from benchy.errors import BenchyError
from benchy.run import run
from benchy.types import compile_schema, validate

LEVELS = {1: "wrong", 2: "weak", 3: "partial", 4: "good", 5: "complete"}
RUBRIC = {"criteria": {
    "accuracy": {"weight": 3, "description": "matches the answer key", "levels": LEVELS},
    "clarity": {"weight": 1, "levels": {0: "unreadable", 1: "clear"}},
}}
GRADER = {"type": "external", "id": "professor"}

OPTIONS = {"a": "string", "b": "string", "c": "string", "d": "string"}

OPEN_ROW = {
    "input": {"question": "¿Por qué cayó el Imperio?"},
    "expected": {"answer": "por sus contradicciones internas"},
}
CHOICE_ROW = {
    "input": {"question": "¿Capital de Francia?", "options": {"a": "Londres", "b": "París", "c": "Roma", "d": "Berlín"}},
    "expected": {"answer": "París"},
}


def mixed(**changes):
    """The mixed exam: open-answer and multiple-choice rows under one program."""
    doc = {
        "ontology_version": "1.1",
        "benchmark": {"task": "answer", "domain": "history", "language": "es"},
        "program": {"input": {"question": "string", "options?": OPTIONS}, "output": {"answer": "string"}},
        "scoring": {
            "weights": {"answer": 1},
            "aggregator": "weighted_mean",
            "field_metrics": {"answer": {
                "metric": "rubric", "params": RUBRIC, "route": {"options": "casefold_strip"},
            }},
            "grader": GRADER,
        },
        "data": {"path": "./exam.jsonl"},
        "ai_system": {"type": "external", "id": "student"},
    }
    doc.update(changes)
    return edit(**doc)


def plain_open():
    """The pre-existing open exam: no optional fields, no route, a plain rubric."""
    return edit(
        ontology_version="1.1",
        benchmark={"task": "answer", "domain": "history", "language": "es"},
        program={"input": {"question": "string"}, "output": {"answer": "string"}},
        scoring={"weights": {"answer": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
        ai_system={"type": "external", "id": "student"},
    )


IR = compile_benchmark(mixed())


def compile_fails(text) -> BenchyError:
    with pytest.raises(BenchyError) as exc:
        compile_benchmark(text)
    return exc.value


# ---------------------------------------------------------------------------
# compiling the mixed exam
# ---------------------------------------------------------------------------

def test_the_optional_input_field_compiles_with_its_marker():
    options = IR["program"]["input"]["fields"]["options"]
    assert options["optional"] is True
    assert options["type"] == "object"
    assert IR["program"]["input"]["fields"]["question"] == {"type": "string"}


def test_the_answer_dimension_carries_its_route():
    (dimension,) = IR["scoring"]["dimensions"]
    assert dimension["metric"] == "rubric"
    assert dimension["route"] == [{"field": "options", "metric": "casefold_strip"}]
    assert IR["scoring"]["grader"] == GRADER


def test_the_mixed_ir_round_trips_through_json():
    assert json.loads(json.dumps(IR)) == IR


def test_a_schema_without_the_marker_compiles_to_no_optional_nodes_and_no_route():
    ir = compile_benchmark(plain_open())
    assert ir["program"]["input"] == {"type": "object", "fields": {"question": {"type": "string"}}}
    assert all("route" not in d for d in ir["scoring"]["dimensions"])


def test_optional_fields_are_rejected_in_the_output_schema():
    error = compile_fails(edit(
        ontology_version="1.1",
        benchmark={"task": "answer", "domain": "history", "language": "es"},
        program={"input": {"question": "string"}, "output": {"answer?": "string"}},
        scoring={"weights": {"answer": 1}, "aggregator": "weighted_mean"},
        data={"path": "./exam.jsonl"},
        ai_system={"type": "external", "id": "student"},
    ))
    assert error.code == "invalid_schema"
    assert error.path[:2] == ["program", "output"]


@pytest.mark.parametrize("bad", ["?", "options??"])
def test_the_optional_marker_is_one_question_mark(bad):
    with pytest.raises(BenchyError) as exc:
        compile_schema({"question": "string", bad: "string"})
    assert exc.value.code == "invalid_schema"


def test_a_field_declared_both_plain_and_optional_is_rejected():
    with pytest.raises(BenchyError) as exc:
        compile_schema({"options": "string", "options?": "string"})
    assert exc.value.code == "invalid_schema"


def test_routing_needs_an_optional_input_field():
    error = compile_fails(mixed(program={
        "input": {"question": "string", "options": OPTIONS},  # present on every row
        "output": {"answer": "string"},
    }))
    assert error.code == "invalid_value"


def test_routing_to_an_unknown_input_field_is_rejected():
    error = compile_fails(mixed(scoring={
        "weights": {"answer": 1}, "aggregator": "weighted_mean",
        "field_metrics": {"answer": {"metric": "rubric", "params": RUBRIC, "route": {"choices": "casefold_strip"}}},
        "grader": GRADER,
    }))
    assert error.code == "unknown_field"


def test_routing_to_the_rubric_is_rejected():
    error = compile_fails(mixed(scoring={
        "weights": {"answer": 1}, "aggregator": "weighted_mean",
        "field_metrics": {"answer": {"metric": "rubric", "params": RUBRIC, "route": {"options": "rubric"}}},
        "grader": GRADER,
    }))
    assert error.code == "invalid_rubric"


def test_routing_to_an_unknown_metric_is_rejected():
    error = compile_fails(mixed(scoring={
        "weights": {"answer": 1}, "aggregator": "weighted_mean",
        "field_metrics": {"answer": {"metric": "rubric", "params": RUBRIC, "route": {"options": "fuzzy"}}},
        "grader": GRADER,
    }))
    assert error.code == "unknown_metric"


def test_a_route_needs_the_input_schema_to_compile():
    scoring = {
        "weights": {"answer": 1}, "aggregator": "weighted_mean",
        "field_metrics": {"answer": {"metric": "rubric", "params": RUBRIC, "route": {"options": "casefold_strip"}}},
        "grader": GRADER,
    }
    with pytest.raises(BenchyError) as exc:
        compile_scoring(scoring, compile_schema({"answer": "string"}))
    assert exc.value.code == "invalid_value"


# ---------------------------------------------------------------------------
# validation: both row shapes under one program
# ---------------------------------------------------------------------------

def rows_validated(rows):
    """Validate both sides of each row the way data.examples does (no artifacts)."""
    return [
        (
            validate(row["input"], IR["program"]["input"], phase="dataset"),
            validate(row["expected"], IR["program"]["output"], phase="dataset"),
        )
        for row in rows
    ]


def test_both_row_shapes_validate_under_one_program():
    (open_in, open_out), (choice_in, choice_out) = rows_validated([OPEN_ROW, CHOICE_ROW])
    assert "options" not in open_in  # absent optional: skipped, not stored as null
    assert choice_in["options"] == CHOICE_ROW["input"]["options"]
    assert open_out == OPEN_ROW["expected"] and choice_out == CHOICE_ROW["expected"]


def test_a_choice_row_missing_an_option_still_fails_missing_field():
    bad = {"input": {"question": "q", "options": {"a": "x", "b": "y", "c": "z"}}, "expected": {"answer": "x"}}
    with pytest.raises(BenchyError) as exc:
        rows_validated([bad])
    assert exc.value.code == "missing_field"
    assert exc.value.path == ["options", "d"]


def test_an_undeclared_input_field_still_fails_extra_field():
    bad = {"input": {"question": "q", "hint": "no va"}, "expected": {"answer": "x"}}
    with pytest.raises(BenchyError) as exc:
        rows_validated([bad])
    assert exc.value.code == "extra_field"


def test_a_null_option_object_is_not_an_absent_one():
    bad = {"input": {"question": "q", "options": None}, "expected": {"answer": "x"}}
    with pytest.raises(BenchyError) as exc:
        rows_validated([bad])
    assert exc.value.code == "wrong_type"


def test_equal_treats_a_missing_optional_field_as_a_difference():
    node = compile_schema({"question": "string", "options?": OPTIONS})
    base = {"question": "q"}
    with_options = {"question": "q", "options": {"a": "1", "b": "2", "c": "3", "d": "4"}}
    assert types.equal(base, dict(base), node)
    assert not types.equal(base, with_options, node)
    assert not types.equal(with_options, base, node)


# ---------------------------------------------------------------------------
# scoring: open rows are judged, choice rows are exact-matched
# ---------------------------------------------------------------------------

class Grader:
    """Full marks for the reference answer, floor for anything else; records calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, graded):
        self.calls.append(graded)
        good = graded["answer"] == graded["reference"]
        return {
            "accuracy": {"grade": 5 if good else 1, "justification": "seen"},
            "clarity": {"grade": 1 if good else 0, "justification": "seen"},
        }


async def execute(tmp_path, student, grader, rows, ir=IR):
    (tmp_path / "exam.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return await run(ir, tmp_path, student, grader=grader)


async def test_a_mixed_exam_scores_open_rows_by_rubric_and_choice_rows_by_exact_match(tmp_path):
    def student(inputs):
        if "options" in inputs:
            return {"answer": " parís "}  # casefold_strip: exact text, any case, any padding
        return {"answer": "por sus contradicciones internas"}

    result = await execute(tmp_path, student, Grader(), [OPEN_ROW, CHOICE_ROW])
    assert result["summary"] == {"examples": 2, "valid": 2, "invalid_outputs": 0, "execution_errors": 0}
    assert result["benchmark_score"] == 1.0
    open_result, choice_result = result["results"]
    assert "criteria" in open_result["field_scores"][0]  # the rubric keeps its grades
    assert "criteria" not in choice_result["field_scores"][0]


async def test_choice_rows_are_never_graded(tmp_path):
    def student(inputs):
        return {"answer": "x"}

    grader = Grader()
    await execute(tmp_path, student, grader, [OPEN_ROW, CHOICE_ROW])
    # One graded baseline (the empty answer) plus one graded prediction: the open
    # row only. The choice row never reaches the grader.
    assert sorted(call["answer"] for call in grader.calls) == ["", "x"]


async def test_a_wrong_answer_scores_zero_under_either_route(tmp_path):
    def student(inputs):
        return {"answer": "Roma"} if "options" in inputs else {"answer": "no lo sé"}

    result = await execute(tmp_path, student, Grader(), [OPEN_ROW, CHOICE_ROW])
    assert result["benchmark_score"] == 0.0
    open_result, choice_result = result["results"]
    assert open_result["field_scores"][0]["score"] == 0.0
    assert choice_result["field_scores"][0]["score"] == 0.0


async def test_the_choice_rows_baseline_is_exact_match_not_the_grader(tmp_path):
    def student(inputs):
        return {"answer": "París"}

    result = await execute(tmp_path, student, Grader(), [CHOICE_ROW])
    (field,) = result["fields"]
    assert field["score"] == 1.0
    assert field["baseline"] == 0.0  # the empty string never casefold-matches "París"
    assert field["counts_as_signal"] is True


async def test_a_pure_open_exam_still_grades_every_row(tmp_path):
    """Regression: without a route, the baseline and every row go through the grader."""
    ir = compile_benchmark(edit(
        ontology_version="1.1",
        benchmark={"task": "answer", "domain": "history", "language": "es"},
        program={"input": {"question": "string"}, "output": {"answer": "string"}},
        scoring={
            "weights": {"answer": 1}, "aggregator": "weighted_mean",
            "field_metrics": {"answer": {"metric": "rubric", "params": RUBRIC}},
            "grader": GRADER,
        },
        data={"path": "./exam.jsonl"},
        ai_system={"type": "external", "id": "student"},
    ))

    def student(inputs):
        return {"answer": "por sus contradicciones internas"}

    grader = Grader()
    result = await execute(tmp_path, student, grader, [OPEN_ROW, OPEN_ROW], ir=ir)
    assert result["benchmark_score"] == 1.0
    # Two baselines (the empty answer per row) plus two graded predictions.
    assert len(grader.calls) == 4
    assert "route" not in ir["scoring"]["dimensions"][0]
