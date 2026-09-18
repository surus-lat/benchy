"""exact_match, contains, regex_match, f1_token, levenshtein_ratio."""

from __future__ import annotations

import pytest

from benchy.core import Scorer
from benchy.scoring.primitives import contains, exact_match, f1_token, levenshtein_ratio, regex_match
from benchy.scoring.registry import parse_scorer


class TestExactMatch:
    def test_is_a_scorer(self):
        assert isinstance(exact_match(), Scorer)

    def test_identical_strings_match(self):
        s = exact_match()
        assert s.fitness("hello", "hello", None) == 1.0

    def test_different_strings_do_not_match(self):
        s = exact_match()
        assert s.fitness("hello", "world", None) == 0.0

    def test_case_insensitive_by_default(self):
        s = exact_match()
        assert s.fitness("Hello", "hello", None) == 1.0

    def test_case_sensitive_when_configured(self):
        s = exact_match(case_insensitive=False)
        assert s.fitness("Hello", "hello", None) == 0.0

    def test_strips_whitespace_by_default(self):
        s = exact_match()
        assert s.fitness("  hello  ", "hello", None) == 1.0

    def test_no_strip_when_configured(self):
        s = exact_match(strip=False)
        assert s.fitness("  hello  ", "hello", None) == 0.0

    def test_coerces_non_strings(self):
        s = exact_match()
        assert s.fitness(42, "42", None) == 1.0

    def test_none_prediction_is_zero(self):
        s = exact_match()
        assert s.fitness(None, "hello", None) == 0.0

    def test_none_expected_is_zero(self):
        s = exact_match()
        assert s.fitness("hello", None, None) == 0.0

    def test_both_none_is_an_exact_match(self):
        # In structured extraction, `null` is often a *correct answer* (e.g.
        # "the caller never stated their name") rather than a missing value.
        # exact_match compares values for literal equality, and None == None
        # is True — so this must score 1.0, not be treated as invalid input.
        # field_wise relies on exactly this to grade nullable fields.
        s = exact_match()
        assert s.fitness(None, None, None) == 1.0

    def test_evaluate_breakdown_carries_normalized_values_and_match_flag(self):
        s = exact_match()
        score = s.evaluate("Hello", "hello", None)
        assert score.value == 1.0
        assert score.breakdown["match"] is True
        assert score.scorer == "exact_match"

    def test_repr_round_trips(self):
        s = exact_match(case_insensitive=False, strip=False)
        assert parse_scorer(repr(s)) == s

    def test_repr_round_trips_with_defaults(self):
        s = exact_match()
        assert parse_scorer(repr(s)) == s

    def test_default_aggregate_reports_fitness(self):
        s = exact_match()
        scores = [s.evaluate("a", "a", None), s.evaluate("a", "b", None)]
        agg = s.aggregate(scores)
        assert agg["fitness"] == pytest.approx(0.5)


class TestContains:
    def test_is_a_scorer(self):
        assert isinstance(contains(), Scorer)

    def test_prediction_containing_expected_matches(self):
        s = contains()
        assert s.fitness("the cat sat on the mat", "cat", None) == 1.0

    def test_prediction_missing_expected_does_not_match(self):
        s = contains()
        assert s.fitness("the dog sat on the mat", "cat", None) == 0.0

    def test_case_insensitive_by_default(self):
        s = contains()
        assert s.fitness("The CAT sat", "cat", None) == 1.0

    def test_case_sensitive_when_configured(self):
        s = contains(case_insensitive=False)
        assert s.fitness("The CAT sat", "cat", None) == 0.0

    def test_none_prediction_is_zero(self):
        s = contains()
        assert s.fitness(None, "cat", None) == 0.0

    def test_none_expected_is_zero(self):
        s = contains()
        assert s.fitness("the cat sat", None, None) == 0.0

    def test_repr_round_trips(self):
        s = contains(case_insensitive=False)
        assert parse_scorer(repr(s)) == s


class TestRegexMatch:
    def test_is_a_scorer(self):
        assert isinstance(regex_match(pattern=r"\d+"), Scorer)

    def test_matches_when_pattern_found_in_prediction(self):
        s = regex_match(pattern=r"^\d{3}-\d{4}$")
        assert s.fitness("555-1234", "unused", None) == 1.0

    def test_no_match_scores_zero(self):
        s = regex_match(pattern=r"^\d{3}-\d{4}$")
        assert s.fitness("not a phone number", "unused", None) == 0.0

    def test_flags_are_honored(self):
        import re

        s = regex_match(pattern="hello", flags=re.IGNORECASE)
        assert s.fitness("HELLO world", "unused", None) == 1.0

    def test_none_pattern_falls_back_to_expected_as_the_regex(self):
        s = regex_match(pattern=None)
        assert s.fitness("555-1234", r"^\d{3}-\d{4}$", None) == 1.0
        assert s.fitness("nope", r"^\d{3}-\d{4}$", None) == 0.0

    def test_none_prediction_is_zero(self):
        s = regex_match(pattern=r"\d+")
        assert s.fitness(None, "unused", None) == 0.0

    def test_invalid_regex_scores_zero_rather_than_raising(self):
        s = regex_match(pattern="[")
        assert s.fitness("anything", "unused", None) == 0.0

    def test_repr_round_trips(self):
        s = regex_match(pattern=r"\d+", flags=0)
        assert parse_scorer(repr(s)) == s

    def test_repr_round_trips_with_none_pattern(self):
        s = regex_match(pattern=None)
        assert parse_scorer(repr(s)) == s


class TestF1Token:
    def test_is_a_scorer(self):
        assert isinstance(f1_token(), Scorer)

    def test_identical_token_sets_score_one(self):
        s = f1_token()
        assert s.fitness("the cat sat", "the cat sat", None) == pytest.approx(1.0)

    def test_partial_overlap_scores_between_zero_and_one(self):
        s = f1_token()
        value = s.fitness("the cat sat on the mat", "the cat sat", None)
        assert 0.0 < value < 1.0

    def test_disjoint_tokens_score_zero(self):
        s = f1_token()
        assert s.fitness("apples oranges", "bananas grapes", None) == 0.0

    def test_none_prediction_scores_zero(self):
        s = f1_token()
        assert s.fitness(None, "the cat sat", None) == 0.0

    def test_none_expected_scores_zero(self):
        s = f1_token()
        assert s.fitness("the cat sat", None, None) == 0.0

    def test_both_empty_scores_zero_not_a_vacuous_match(self):
        s = f1_token()
        assert s.fitness("", "", None) == 0.0

    def test_expected_as_list_of_references_takes_the_max(self):
        s = f1_token()
        value = s.fitness("the cat sat", ["a totally different sentence", "the cat sat"], None)
        assert value == pytest.approx(1.0)

    def test_breakdown_carries_precision_and_recall(self):
        s = f1_token()
        score = s.evaluate("the cat sat", "the cat sat", None)
        assert score.breakdown["precision"] == pytest.approx(1.0)
        assert score.breakdown["recall"] == pytest.approx(1.0)

    def test_repr_round_trips(self):
        s = f1_token()
        assert parse_scorer(repr(s)) == s


class TestLevenshteinRatio:
    def test_is_a_scorer(self):
        assert isinstance(levenshtein_ratio(), Scorer)

    def test_identical_strings_score_one(self):
        s = levenshtein_ratio()
        assert s.fitness("hello", "hello", None) == pytest.approx(1.0)

    def test_completely_different_strings_score_low(self):
        s = levenshtein_ratio()
        # Equal-length total-mismatch strings bottom out at exactly 0.5 (pure
        # substitution); use a length mismatch too so the ratio drops clearly
        # below that floor.
        assert s.fitness("hello", "zzzzzzzzzzzzzzzzzzzz", None) < 0.5

    def test_close_strings_score_high(self):
        s = levenshtein_ratio()
        assert s.fitness("hello", "hallo", None) > 0.7

    def test_none_prediction_scores_zero(self):
        s = levenshtein_ratio()
        assert s.fitness(None, "hello", None) == 0.0

    def test_none_expected_scores_zero(self):
        s = levenshtein_ratio()
        assert s.fitness("hello", None, None) == 0.0

    def test_both_empty_scores_one(self):
        s = levenshtein_ratio()
        assert s.fitness("", "", None) == 1.0

    def test_repr_round_trips(self):
        s = levenshtein_ratio()
        assert parse_scorer(repr(s)) == s
