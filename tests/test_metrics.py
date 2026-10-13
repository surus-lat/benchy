"""Field metrics: the closed registry, `scoring.field_metrics`, and baselines.

Covers the registry's behavior (unit), the compiler's validation of
`scoring.field_metrics` (unknown_metric / enum_unsafe_metric / param bounds), the
structural compatibility of a benchmark without `field_metrics`, and the run's
per-field baseline + anti-trivial guard.
"""

from __future__ import annotations

import json

import pytest
from conftest import edit

from benchy import metrics
from benchy.compiler import compile_benchmark
from benchy.errors import BenchyError
from benchy.run import run

# ---------------------------------------------------------------------------
# the registry: fold parity and per-metric behavior
# ---------------------------------------------------------------------------

def test_fold_strips_accents_case_and_whitespace():
    assert metrics._fold("  Café   CONEJAÑÓ ") == "cafe conejano"


def test_tokens_are_alphanumeric_and_order_free():
    assert metrics._tokens("ACME S.A. 2026") == frozenset({"acme", "s", "a", "2026"})


def test_token_set_f1_scores_reordered_tokens_where_exact_does_not():
    # The case the registry exists for: a correct extraction with reordered tokens.
    assert metrics.score("token_set_f1", "ACME SA", "sa acme") == 1.0
    assert metrics.score("exact", "ACME SA", "sa acme") == 0.0


def test_token_set_f1_is_continuous():
    assert metrics.score("token_set_f1", "acme sa", "acme") == pytest.approx(2 / 3)
    assert metrics.score("token_set_f1", "", "acme") == 0.0
    assert metrics.score("token_set_f1", "", "") == 1.0


def test_casefold_strip_matches_case_and_padding_only():
    assert metrics.score("casefold_strip", "  Acme ", "acme") == 1.0
    assert metrics.score("casefold_strip", "acme sa", "acme") == 0.0
    assert metrics.score("casefold_strip", 1, 1) == 1.0  # non-strings: plain equality


def test_date_flexible_parses_common_spellings():
    assert metrics.score("date_flexible", "12/3/26", "2026-03-12") == 1.0
    assert metrics.score("date_flexible", "12 de marzo de 2026", "2026-03-12") == 1.0
    assert metrics.score("date_flexible", "2026-03-13", "2026-03-12") == 0.0
    assert metrics.score("date_flexible", "not a date", "2026-03-12") == 0.0


def test_numeric_tolerance_uses_the_tolerance_param():
    assert metrics.score("numeric_tolerance", "121,50", 121.0, {"tolerance": 0.5}) == 1.0
    assert metrics.score("numeric_tolerance", 122.0, 121.0, {"tolerance": 0.5}) == 0.0
    assert metrics.score("numeric_tolerance", 121.0, 121.0) == 1.0
    assert metrics.score("numeric_tolerance", "n/a", 121.0, {"tolerance": 0.5}) == 0.0


def test_span_recall_counts_expected_tokens_found_in_the_prediction():
    assert metrics.score("span_recall", "el total es 120 pesos", "total 120") == 1.0
    assert metrics.score("span_recall", "el total es otro", "total 120") == 0.5
    assert metrics.score("span_recall", "anything", "") == 0.0


def test_set_f1_compares_collections_and_token_sets():
    assert metrics.score("set_f1", ["a", "b"], ["b", "a"]) == 1.0
    assert metrics.score("set_f1", ["a"], ["a", "b"]) == pytest.approx(2 / 3)
    assert metrics.score("set_f1", "x y", "y x") == 1.0


def test_unknown_metric_name_is_rejected_by_the_registry():
    with pytest.raises(ValueError, match="unknown metric"):
        metrics.score("always_right", "a", "b")


def test_catalog_lists_params_bounds_and_enum_safety():
    catalog = metrics.catalog()
    assert set(catalog) == set(metrics.METRICS)
    assert catalog["exact"]["enum_safe"] is True
    assert catalog["span_recall"]["enum_safe"] is False
    assert catalog["numeric_tolerance"]["params"]["tolerance"] == [0.0, "unbounded"]


# ---------------------------------------------------------------------------
# the compiler validates scoring.field_metrics against the closed registry
# ---------------------------------------------------------------------------

FIELD_METRICS_YAML = {
    "weights": {"supplier": 1, "total": 3},
    "aggregator": "weighted_mean",
    "field_metrics": {"supplier": {"metric": "token_set_f1"}},
}


def compile_with(scoring, output=None):
    return compile_benchmark(edit(
        program={"input": {"text": "string"},
                 "output": output or {"supplier": "string", "total": "float"}},
        scoring=scoring,
        data={"path": "./exam.jsonl"},
    ))


def test_field_metrics_stamps_the_dimension_with_metric_and_params():
    ir = compile_with({
        "weights": {"supplier": 1, "total": 3},
        "aggregator": "weighted_mean",
        "field_metrics": {
            "supplier": {"metric": "token_set_f1"},
            "total": {"metric": "numeric_tolerance", "params": {"tolerance": 0.5}},
        },
    })
    (supplier, total) = ir["scoring"]["dimensions"]
    assert supplier["metric"] == "token_set_f1"
    assert supplier["params"] == {}
    assert total["metric"] == "numeric_tolerance"
    assert total["params"] == {"tolerance": 0.5}


def test_a_benchmark_without_field_metrics_compiles_to_exactly_the_same_ir():
    plain = compile_with({"weights": {"supplier": 1, "total": 3}, "aggregator": "weighted_mean"})
    assert plain["scoring"]["dimensions"] == [
        {"path": ["supplier"], "weight": 1.0},
        {"path": ["total"], "weight": 3.0},
    ]
    assert "signal_epsilon" not in plain["scoring"]


def test_unknown_metric_is_rejected_with_the_field_path():
    with pytest.raises(BenchyError) as exc:
        compile_with({**FIELD_METRICS_YAML,
                      "field_metrics": {"supplier": {"metric": "always_right"}}})
    assert exc.value.code == "unknown_metric"
    assert exc.value.path == ["scoring", "field_metrics", "supplier"]


def test_a_non_enum_safe_metric_over_an_enum_field_is_rejected():
    with pytest.raises(BenchyError) as exc:
        compile_with(
            {"weights": {"cat": 1}, "aggregator": "weighted_mean",
             "field_metrics": {"cat": {"metric": "span_recall"}}},
            output={"cat": {"enum": ["a", "b"]}},
        )
    assert exc.value.code == "enum_unsafe_metric"
    assert exc.value.path == ["scoring", "field_metrics", "cat"]


def test_an_enum_safe_metric_over_an_enum_field_compiles():
    ir = compile_with(
        {"weights": {"cat": 1}, "aggregator": "weighted_mean",
         "field_metrics": {"cat": {"metric": "casefold_strip"}}},
        output={"cat": {"enum": ["a", "b"]}},
    )
    assert ir["scoring"]["dimensions"][0]["metric"] == "casefold_strip"


def test_field_metrics_key_must_be_an_output_leaf():
    with pytest.raises(BenchyError) as exc:
        compile_with({**FIELD_METRICS_YAML,
                      "field_metrics": {"nope": {"metric": "exact"}}})
    assert exc.value.code == "unknown_field"


def test_unknown_param_is_rejected():
    with pytest.raises(BenchyError) as exc:
        compile_with({**FIELD_METRICS_YAML,
                      "field_metrics": {"supplier": {"metric": "token_set_f1", "params": {"foo": 1}}}})
    assert exc.value.code == "unknown_param"


def test_param_out_of_range_is_rejected():
    with pytest.raises(BenchyError) as exc:
        compile_with({**FIELD_METRICS_YAML,
                      "field_metrics": {"total": {"metric": "numeric_tolerance",
                                                  "params": {"tolerance": -0.5}}}})
    assert exc.value.code == "param_out_of_range"


def test_field_metrics_rejects_unknown_keys_in_an_entry():
    with pytest.raises(BenchyError) as exc:
        compile_with({**FIELD_METRICS_YAML,
                      "field_metrics": {"supplier": {"metric": "exact", "because": "yes"}}})
    assert exc.value.code == "unknown_key"


@pytest.mark.parametrize("epsilon", [0, -0.01, 0.2, "0.01"])
def test_signal_epsilon_outside_the_sealed_range_is_rejected(epsilon):
    with pytest.raises(BenchyError) as exc:
        compile_with({"weights": {"supplier": 1, "total": 3}, "aggregator": "weighted_mean",
                      "signal_epsilon": epsilon})
    assert exc.value.code == "invalid_value"


def test_a_valid_signal_epsilon_is_carried_into_the_ir():
    ir = compile_with({"weights": {"supplier": 1, "total": 3}, "aggregator": "weighted_mean",
                       "signal_epsilon": 0.05})
    assert ir["scoring"]["signal_epsilon"] == 0.05


# ---------------------------------------------------------------------------
# the run: per-field baselines and the anti-trivial guard
# ---------------------------------------------------------------------------

def workspace(tmp_path, *expected):
    rows = [{"input": {"text": f"row-{i}"}, "expected": e} for i, e in enumerate(expected)]
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "exam.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path


async def test_the_report_publishes_score_baseline_and_signal_per_field(tmp_path):
    ir = compile_with({"weights": {"supplier": 1, "total": 3}, "aggregator": "weighted_mean"})
    good = {"supplier": "ACME", "total": 121.0}
    result = await run(ir, workspace(tmp_path, good), lambda _: dict(good))
    (supplier, total) = result["fields"]
    assert set(supplier) == {"path", "metric", "weight", "score", "baseline", "signal", "counts_as_signal"}
    assert supplier["path"] == ["supplier"]
    assert supplier["metric"] == "exact"
    assert supplier["score"] == 1.0
    assert supplier["baseline"] == 0.0  # the empty string never matches "ACME"
    assert supplier["signal"] == 1.0
    assert supplier["counts_as_signal"] is True
    assert total["baseline"] == 1.0     # the exam's mean *is* 121.0
    assert total["counts_as_signal"] is False  # score 1.0 < baseline 1.0 + epsilon


async def test_a_score_equal_to_the_majority_baseline_is_not_signal(tmp_path):
    ir = compile_with({"weights": {"cat": 1}, "aggregator": "weighted_mean"},
                      output={"cat": {"enum": ["a", "b"]}})
    expected = [{"cat": "a"}, {"cat": "a"}, {"cat": "b"}]
    # A system that always answers the majority label: score == baseline == 2/3.
    result = await run(ir, workspace(tmp_path, *expected), lambda _: {"cat": "a"})
    assert result["benchmark_score"] == pytest.approx(2 / 3)
    (cat,) = result["fields"]
    assert cat["baseline"] == pytest.approx(2 / 3)
    assert cat["signal"] == pytest.approx(0.0)
    assert cat["counts_as_signal"] is False


async def test_field_metrics_change_what_a_field_scores(tmp_path):
    ir = compile_with(FIELD_METRICS_YAML)
    expected = {"supplier": "ACME SA", "total": 121.0}
    # Reordered tokens: exact would miss, token_set_f1 scores 1.
    result = await run(ir, workspace(tmp_path, expected),
                       lambda _: {"supplier": "sa acme", "total": 121.0})
    assert result["benchmark_score"] == 1.0


async def test_a_zero_weight_field_with_a_metric_reports_but_does_not_score(tmp_path):
    ir = compile_with({"weights": {"note": 0, "total": 1}, "aggregator": "weighted_mean",
                       "field_metrics": {"note": {"metric": "span_recall"}}},
                      output={"note": "string", "total": "float"})
    expected = {"note": "el total es 121", "total": 1.0}
    result = await run(ir, workspace(tmp_path, expected),
                       lambda _: {"note": "nada que ver", "total": 1.0})
    assert result["results"][0]["score"] == 1.0  # the failed weight-0 field does not score
    (note, _) = result["fields"]
    assert note["weight"] == 0.0
    assert note["score"] == 0.0  # but it is still reported, and it is not signal


async def test_the_epsilon_comes_from_the_ir_when_declared(tmp_path):
    scoring = {"weights": {"cat": 1}, "aggregator": "weighted_mean"}
    expected = [{"cat": "a"}] * 80 + [{"cat": "b"}] * 20
    # A system 0.03 above the majority baseline: signal at the default epsilon
    # (0.01), not at the declared 0.05.
    answers = iter([{"cat": "a"}] * 80 + [{"cat": "b"}] * 3 + [{"cat": "a"}] * 17)

    default_ir = compile_with(scoring, output={"cat": {"enum": ["a", "b"]}})
    strict_ir = compile_with({**scoring, "signal_epsilon": 0.05},
                             output={"cat": {"enum": ["a", "b"]}})
    at_default = await run(default_ir, workspace(tmp_path / "d", *expected), lambda _: next(answers))
    answers = iter([{"cat": "a"}] * 80 + [{"cat": "b"}] * 3 + [{"cat": "a"}] * 17)
    at_declared = await run(strict_ir, workspace(tmp_path / "s", *expected), lambda _: next(answers))

    for result, expected_flag in ((at_default, True), (at_declared, False)):
        (field,) = result["fields"]
        assert field["score"] == pytest.approx(0.83)
        assert field["baseline"] == pytest.approx(0.8)
        assert field["counts_as_signal"] is expected_flag


async def test_fields_report_covers_failed_examples_in_the_denominator(tmp_path):
    ir = compile_with({"weights": {"supplier": 1}, "aggregator": "weighted_mean"},
                      output={"supplier": "string"})
    good = {"supplier": "ACME"}

    def half_bad(_):
        half_bad.n += 1
        return dict(good) if half_bad.n == 1 else {"wrong": "shape"}
    half_bad.n = 0

    result = await run(ir, workspace(tmp_path, good, good), half_bad)
    (supplier,) = result["fields"]
    assert supplier["score"] == 0.5  # one valid hit, one zero contribution
