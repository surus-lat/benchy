"""A constant prediction must not read as a measurement.

The report's shape is a pinned contract, so this diagnostic is a warning on stderr and
never a new key in the report: an adapter that ignores its input (or reads the wrong input
key) returns the same output for every example and would otherwise score 100% valid.

The same non-measurement exists one level down: a single field can be constant, or an
exam's expected value for a single field can be constant, while the whole object varies.
"""
from __future__ import annotations

import json

from conftest import edit

from benchy.cli import _warn_if_degenerate, _warn_if_uninformative_field
from benchy.compiler import compile_benchmark
from benchy.run import run


def _result(predictions):
    return {"results": [{"status": "valid", "prediction": item} for item in predictions]}


def test_a_constant_predictor_is_reported(capsys):
    _warn_if_degenerate(_result([{"label": "yes"}] * 3))
    captured = capsys.readouterr()
    assert "degenerate_constant_output" in captured.err
    assert captured.out == ""


def test_varying_predictions_are_silent(capsys):
    _warn_if_degenerate(_result([{"label": "yes"}, {"label": "no"}]))
    assert capsys.readouterr().err == ""


def test_a_single_example_is_not_degenerate(capsys):
    _warn_if_degenerate(_result([{"label": "yes"}]))
    assert capsys.readouterr().err == ""


def test_invalid_outputs_are_not_counted_as_predictions(capsys):
    result = _result([{"label": "yes"}])
    result["results"].append({"status": "invalid_output", "prediction": {"label": "no"}})
    _warn_if_degenerate(result)
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# The field level: the field's score looks like performance and measures nothing
# ---------------------------------------------------------------------------

FIELD_DOC = edit(
    program={"input": {"text": "string"}, "output": {"supplier": "string", "total": "float"}},
    scoring={"weights": {"supplier": 1, "total": 3}, "aggregator": "weighted_mean"},
    data={"path": "./exam.jsonl"},
)
FIELD_IR = compile_benchmark(FIELD_DOC)


async def _field_warnings(tmp_path, expected, predictions):
    """Run the exam and emit both diagnostics, exactly as `benchy run` does."""
    (tmp_path / "exam.jsonl").write_text(
        "\n".join(
            json.dumps({"input": {"text": f"row-{index}"}, "expected": output})
            for index, output in enumerate(expected)
        )
    )
    queue = list(predictions)
    result = await run(FIELD_IR, tmp_path, lambda _: queue.pop(0))
    _warn_if_degenerate(result)
    _warn_if_uninformative_field(result, FIELD_IR, tmp_path)
    return result


async def test_a_field_the_exam_never_varies_is_reported(tmp_path, capsys):
    """Any predictor that emits that constant scores 1.0 on the field, so it is not a field score."""
    expected = [{"supplier": "ACME", "total": 1.0}, {"supplier": "ACME", "total": 2.0},
                {"supplier": "ACME", "total": 3.0}]
    predictions = [{"supplier": "ACME", "total": 1.0}, {"supplier": "OTHER", "total": 2.0},
                   {"supplier": "ACME", "total": 3.0}]
    await _field_warnings(tmp_path, expected, predictions)
    err = capsys.readouterr().err
    assert "degenerate_exam_field" in err
    assert "supplier" in err
    assert '"ACME" 3/3' in err


async def test_a_constant_field_is_reported_even_when_the_object_varies(tmp_path, capsys):
    """The whole-object check is silent here: only the field is constant."""
    expected = [{"supplier": "ACME", "total": 1.0}, {"supplier": "BETA", "total": 2.0},
                {"supplier": "ACME", "total": 3.0}]
    predictions = [{"supplier": "ACME", "total": 1.0}, {"supplier": "ACME", "total": 2.0},
                   {"supplier": "ACME", "total": 3.0}]
    await _field_warnings(tmp_path, expected, predictions)
    err = capsys.readouterr().err
    assert "degenerate_constant_output" not in err
    assert "degenerate_constant_field" in err
    assert "supplier" in err
    assert '"ACME" 2/3, "BETA" 1/3' in err


async def test_a_perfect_field_shows_the_baseline_it_has_to_beat(tmp_path, capsys):
    expected = [{"supplier": "ACME", "total": 1.0}, {"supplier": "BETA", "total": 2.0},
                {"supplier": "BETA", "total": 3.0}]
    await _field_warnings(tmp_path, expected, expected)
    err = capsys.readouterr().err
    assert "field_score_baseline" in err
    assert "supplier" in err
    assert "0.667" in err


async def test_varying_predictions_over_a_wide_alphabet_are_silent(tmp_path, capsys):
    expected = [{"supplier": f"S{index}", "total": float(index)} for index in range(4)]
    predictions = [{"supplier": "S0", "total": 0.0}, {"supplier": "S1", "total": 1.0},
                   {"supplier": "S2", "total": 2.0}, {"supplier": "OTHER", "total": 9.0}]
    await _field_warnings(tmp_path, expected, predictions)
    assert capsys.readouterr().err == ""


async def test_a_single_valid_example_is_not_degenerate(tmp_path, capsys):
    await _field_warnings(tmp_path, [{"supplier": "ACME", "total": 1.0}],
                          [{"supplier": "ACME", "total": 1.0}])
    assert capsys.readouterr().err == ""
