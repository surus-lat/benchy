"""A constant prediction must not read as a measurement.

The report's shape is a pinned contract, so this diagnostic is a warning on stderr and
never a new key in the report: an adapter that ignores its input (or reads the wrong input
key) returns the same output for every example and would otherwise score 100% valid.
"""
from benchy.cli import _warn_if_degenerate


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
