"""binary, threshold, restrict, mean, weighted_sum, invert, clamp.

Every transform here takes one or more Scorers and returns a new Scorer —
"operators build new scorers from old" per the module's design mandate.
"""

from __future__ import annotations

import pytest

from benchy.core import Sample, Scorer
from benchy.scoring.primitives import exact_match, f1_token, numeric_close
from benchy.scoring.registry import parse_scorer
from benchy.scoring.structural import field_wise
from benchy.scoring.transforms import binary, clamp, invert, mean, restrict, threshold, weighted_sum


class TestBinary:
    def test_is_a_scorer(self):
        assert isinstance(binary(exact_match()), Scorer)

    def test_passes_through_a_perfect_inner_score(self):
        s = binary(exact_match())
        assert s.fitness("a", "a", None) == 1.0

    def test_collapses_a_below_cutoff_score_to_zero(self):
        s = binary(f1_token(), cutoff=0.9)
        value = s.fitness("the cat sat on the mat", "the cat sat", None)  # partial overlap, not 1.0
        assert value == 0.0

    def test_default_cutoff_is_one_half(self):
        s = binary(f1_token())
        high = s.fitness("the cat sat", "the cat sat", None)
        assert high == 1.0

    def test_repr_round_trips(self):
        s = binary(exact_match(), cutoff=0.75)
        assert parse_scorer(repr(s)) == s

    def test_repr_round_trips_nested_field_wise(self):
        s = binary(field_wise(fields=("a", "b"), per_field=exact_match()))
        assert parse_scorer(repr(s)) == s


class TestThreshold:
    def test_is_a_scorer(self):
        assert isinstance(threshold(f1_token(), cutoff=0.5), Scorer)

    def test_above_cutoff_keeps_the_graded_value(self):
        s = threshold(f1_token(), cutoff=0.3)
        value = s.fitness("the cat sat on the mat", "the cat sat", None)
        assert 0.0 < value < 1.0

    def test_below_cutoff_floors_to_zero_by_default(self):
        s = threshold(f1_token(), cutoff=0.99)
        value = s.fitness("the cat sat on the mat", "the cat sat", None)
        assert value == 0.0

    def test_custom_floor_value(self):
        s = threshold(f1_token(), cutoff=0.99, floor=0.1)
        value = s.fitness("the cat sat on the mat", "the cat sat", None)
        assert value == 0.1

    def test_repr_round_trips(self):
        s = threshold(exact_match(), cutoff=0.4, floor=0.05)
        assert parse_scorer(repr(s)) == s


class TestRestrict:
    def test_is_a_scorer(self):
        inner = field_wise(fields=("a", "b", "c"), per_field=exact_match())
        assert isinstance(restrict(inner, fields=("a",)), Scorer)

    def test_projects_prediction_and_expected_to_the_given_fields(self):
        inner = field_wise(fields=None, per_field=exact_match())
        restricted = restrict(inner, fields=("vendor",))
        pred = {"vendor": "Acme", "total": "999"}
        exp = {"vendor": "Acme", "total": "100"}
        # "total" differs but is excluded by restrict, so the score is perfect.
        assert restricted.fitness(pred, exp, None) == 1.0

    def test_non_mapping_inputs_pass_through_unchanged(self):
        s = restrict(exact_match(), fields=("x",))
        assert s.fitness("hello", "hello", None) == 1.0

    def test_repr_round_trips(self):
        inner = field_wise(fields=None, per_field=exact_match())
        s = restrict(inner, fields=("vendor", "total"))
        assert parse_scorer(repr(s)) == s


class TestMean:
    def test_is_a_scorer(self):
        assert isinstance(mean(exact_match(), f1_token()), Scorer)

    def test_averages_component_fitness(self):
        s = mean(exact_match(), f1_token())
        # exact_match("a b", "a b") = 1.0 (identical); f1_token same pair = 1.0
        assert s.fitness("a b", "a b", None) == pytest.approx(1.0)

    def test_disagreeing_components_average_between_them(self):
        s = mean(exact_match(), f1_token())
        # exact_match("a x", "a b") = 0.0; f1_token("a x", "a b") = 0.5 (1/2 overlap)
        value = s.fitness("a x", "a b", None)
        assert value == pytest.approx(0.25)

    def test_breakdown_lists_each_component(self):
        s = mean(exact_match(), f1_token())
        score = s.evaluate("a b", "a b", None)
        assert len(score.breakdown["components"]) == 2

    def test_repr_round_trips(self):
        s = mean(exact_match(), f1_token())
        assert parse_scorer(repr(s)) == s


class TestWeightedSum:
    def test_is_a_scorer(self):
        s = weighted_sum({"correctness": exact_match(), "style": f1_token()}, weights={"correctness": 0.8, "style": 0.2})
        assert isinstance(s, Scorer)

    def test_weights_bias_the_combination(self):
        s = weighted_sum(
            {"correctness": exact_match(), "style": f1_token()},
            weights={"correctness": 0.9, "style": 0.1},
        )
        # exact_match=0.0 (mismatch), f1_token=0.5 (partial overlap)
        value = s.fitness("a x", "a b", None)
        assert value == pytest.approx(0.9 * 0.0 + 0.1 * 0.5)

    def test_breakdown_is_keyed_by_component_name(self):
        s = weighted_sum({"correctness": exact_match(), "style": f1_token()}, weights={"correctness": 1, "style": 1})
        score = s.evaluate("a b", "a b", None)
        assert set(score.breakdown["components"].keys()) == {"correctness", "style"}

    def test_repr_round_trips(self):
        s = weighted_sum(
            {"correctness": exact_match(), "style": f1_token()},
            weights={"correctness": 0.8, "style": 0.2},
        )
        assert parse_scorer(repr(s)) == s


class TestInvert:
    def test_is_a_scorer(self):
        assert isinstance(invert(exact_match()), Scorer)

    def test_flips_a_perfect_score_to_zero(self):
        s = invert(exact_match())
        assert s.fitness("a", "a", None) == 0.0

    def test_flips_a_zero_score_to_one(self):
        s = invert(exact_match())
        assert s.fitness("a", "b", None) == 1.0

    def test_repr_round_trips(self):
        s = invert(exact_match())
        assert parse_scorer(repr(s)) == s


class TestClamp:
    def test_is_a_scorer(self):
        assert isinstance(clamp(numeric_close()), Scorer)

    def test_clamps_values_above_hi(self):
        s = clamp(exact_match(), lo=0.0, hi=0.8)
        assert s.fitness("a", "a", None) == 0.8

    def test_clamps_values_below_lo(self):
        s = clamp(exact_match(), lo=0.2, hi=1.0)
        assert s.fitness("a", "b", None) == 0.2

    def test_values_inside_range_pass_through(self):
        s = clamp(numeric_close(), lo=0.0, hi=1.0)
        value = s.fitness(50.0, 100.0, None)
        assert 0.0 <= value <= 1.0

    def test_repr_round_trips(self):
        s = clamp(exact_match(), lo=0.1, hi=0.9)
        assert parse_scorer(repr(s)) == s


class TestAggregateAlwaysCarriesFitness:
    """Every transform must produce an aggregate()["fitness"] in [0, 1] —
    the engine reads Report.fitness straight out of it with no rescaling."""

    @pytest.mark.parametrize(
        "scorer",
        [
            binary(exact_match()),
            threshold(f1_token(), cutoff=0.5),
            restrict(field_wise(fields=None, per_field=exact_match()), fields=("a",)),
            mean(exact_match(), f1_token()),
            weighted_sum({"a": exact_match()}, weights={"a": 1.0}),
            invert(exact_match()),
            clamp(exact_match()),
        ],
    )
    def test_aggregate_has_fitness_in_unit_interval(self, scorer):
        scores = [scorer.evaluate("a", "a", None), scorer.evaluate("a", "b", None)]
        agg = scorer.aggregate(scores)
        assert "fitness" in agg
        assert 0.0 <= agg["fitness"] <= 1.0
