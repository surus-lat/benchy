"""numeric_close, mse, pearson.

`mse` is an error-family metric: lower raw MSE is better, but `fitness` and
`Score.value` must invert so higher is always better everywhere in this
package. We assert the inversion formula (`1 / (1 + mse)`) directly, since
getting it backwards would silently break every optimizer downstream.
"""

from __future__ import annotations

import pytest

from benchy.core import Sample, Score, Scorer
from benchy.scoring.primitives import mse, numeric_close, pearson
from benchy.scoring.registry import parse_scorer


class TestNumericClose:
    def test_is_a_scorer(self):
        assert isinstance(numeric_close(), Scorer)

    def test_exact_match_scores_one(self):
        s = numeric_close()
        assert s.fitness(3.0, 3.0, None) == 1.0

    def test_within_tolerance_scores_one(self):
        s = numeric_close(rel_tol=0.01)
        assert s.fitness(100.5, 100.0, None) == 1.0

    def test_outside_tolerance_scores_less_than_one(self):
        s = numeric_close(rel_tol=0.001)
        assert s.fitness(50.0, 100.0, None) < 1.0

    def test_wildly_off_values_score_near_zero(self):
        s = numeric_close()
        assert s.fitness(-1000.0, 100.0, None) == pytest.approx(0.0, abs=1e-6)

    def test_value_never_goes_negative(self):
        s = numeric_close()
        assert s.fitness(1e9, 1.0, None) >= 0.0

    def test_non_numeric_prediction_scores_zero(self):
        s = numeric_close()
        assert s.fitness("not a number", 3.0, None) == 0.0

    def test_none_prediction_scores_zero(self):
        s = numeric_close()
        assert s.fitness(None, 3.0, None) == 0.0

    def test_none_expected_scores_zero(self):
        s = numeric_close()
        assert s.fitness(3.0, None, None) == 0.0

    def test_both_none_is_an_exact_match(self):
        # Same null-as-valid-answer convention as exact_match: a field whose
        # correct value is documented as null, predicted as null, is correct.
        s = numeric_close()
        assert s.fitness(None, None, None) == 1.0

    def test_zero_expected_uses_absolute_tolerance(self):
        s = numeric_close(abs_tol=0.01)
        assert s.fitness(0.005, 0.0, None) == 1.0

    def test_breakdown_reports_diff_and_within_tolerance(self):
        s = numeric_close()
        score = s.evaluate(3.0, 3.0, None)
        assert score.breakdown["within_tolerance"] is True
        assert score.breakdown["diff"] == pytest.approx(0.0)

    def test_repr_round_trips(self):
        s = numeric_close(rel_tol=0.05, abs_tol=0.1)
        assert parse_scorer(repr(s)) == s


class TestMeanSquaredError:
    def test_is_a_scorer(self):
        assert isinstance(mse(), Scorer)

    def test_zero_error_is_perfect_fitness(self):
        s = mse()
        assert s.fitness(5.0, 5.0, None) == 1.0

    def test_fitness_is_the_inversion_one_over_one_plus_mse(self):
        s = mse()
        raw = (3.0 - 5.0) ** 2  # 4.0
        expected_fitness = 1.0 / (1.0 + raw)
        assert s.fitness(3.0, 5.0, None) == pytest.approx(expected_fitness)

    def test_fitness_stays_in_unit_interval_for_huge_errors(self):
        s = mse()
        value = s.fitness(1_000_000.0, 0.0, None)
        assert 0.0 <= value <= 1.0

    def test_breakdown_carries_the_raw_mse_uninverted(self):
        s = mse()
        score = s.evaluate(3.0, 5.0, None)
        assert score.breakdown["mse"] == pytest.approx(4.0)

    def test_non_numeric_inputs_score_zero_fitness(self):
        s = mse()
        assert s.fitness("nope", 5.0, None) == 0.0

    def test_none_inputs_score_zero_fitness(self):
        s = mse()
        assert s.fitness(None, 5.0, None) == 0.0

    def test_both_none_is_a_perfect_fitness(self):
        s = mse()
        assert s.fitness(None, None, None) == 1.0

    def test_aggregate_reports_both_fitness_and_raw_mse(self):
        s = mse()
        scores = [s.evaluate(5.0, 5.0, None), s.evaluate(3.0, 5.0, None)]
        agg = s.aggregate(scores)
        assert "fitness" in agg
        assert agg["mse"] == pytest.approx((0.0 + 4.0) / 2)

    def test_repr_round_trips(self):
        s = mse()
        assert parse_scorer(repr(s)) == s


class TestPearson:
    """Pearson correlation is fundamentally a run-level statistic: it isn't
    defined for a single (prediction, expected) pair. `evaluate` therefore
    returns a *validity* indicator per sample (was this pair usable numeric
    data?) and stashes the raw pair in `breakdown` for `aggregate` to compute
    the actual correlation coefficient across the whole run."""

    def test_is_a_scorer(self):
        assert isinstance(pearson(), Scorer)

    def test_valid_numeric_pair_scores_one_on_evaluate(self):
        s = pearson()
        assert s.evaluate(1.0, 2.0, None).value == 1.0

    def test_invalid_pair_scores_zero_on_evaluate(self):
        s = pearson()
        assert s.evaluate("nope", 2.0, None).value == 0.0

    def test_none_pair_scores_zero_on_evaluate(self):
        s = pearson()
        assert s.evaluate(None, 2.0, None).value == 0.0

    def test_breakdown_carries_the_raw_pair(self):
        s = pearson()
        score = s.evaluate(1.0, 2.0, None)
        assert score.breakdown["prediction"] == 1.0
        assert score.breakdown["expected"] == 2.0

    def test_aggregate_computes_perfect_positive_correlation(self):
        s = pearson()
        pairs = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
        scores = [s.evaluate(p, e, None) for p, e in pairs]
        agg = s.aggregate(scores)
        assert agg["pearson"] == pytest.approx(1.0)

    def test_aggregate_computes_perfect_negative_correlation(self):
        s = pearson()
        pairs = [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)]
        scores = [s.evaluate(p, e, None) for p, e in pairs]
        agg = s.aggregate(scores)
        assert agg["pearson"] == pytest.approx(-1.0)

    def test_aggregate_fitness_maps_correlation_into_unit_interval(self):
        s = pearson()
        pairs = [(1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.0)]
        scores = [s.evaluate(p, e, None) for p, e in pairs]
        agg = s.aggregate(scores)
        # r == -1.0 maps to fitness 0.0 via (r + 1) / 2
        assert agg["fitness"] == pytest.approx(0.0)

    def test_aggregate_with_fewer_than_two_valid_pairs_is_neutral(self):
        s = pearson()
        scores = [s.evaluate(1.0, 1.0, None)]
        agg = s.aggregate(scores)
        assert agg["pearson"] == 0.0

    def test_aggregate_ignores_invalid_pairs(self):
        s = pearson()
        scores = [
            s.evaluate(1.0, 1.0, None),
            s.evaluate(2.0, 2.0, None),
            s.evaluate("garbage", "also garbage", None),
        ]
        agg = s.aggregate(scores)
        assert agg["pearson"] == pytest.approx(1.0)

    def test_repr_round_trips(self):
        s = pearson()
        assert parse_scorer(repr(s)) == s
