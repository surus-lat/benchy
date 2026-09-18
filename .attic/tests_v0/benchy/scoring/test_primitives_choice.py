"""multiple_choice_accuracy.

Contract: `sample.meta["choices"]` carries the option texts for that sample
(optionally `"choice_labels"` and `"label_to_index"`). `expected` is the
correct 0-based index or a label string. `prediction` is whatever the model
said — an int, a label letter, JSON, or freeform text naming a choice.
"""

from __future__ import annotations

import pytest

from benchy.core import Sample, Scorer
from benchy.scoring.primitives import multiple_choice_accuracy
from benchy.scoring.registry import parse_scorer


def _sample(choices, **extra_meta) -> Sample:
    return Sample(id="s1", input={}, meta={"choices": choices, **extra_meta})


class TestMultipleChoiceAccuracy:
    def test_is_a_scorer(self):
        assert isinstance(multiple_choice_accuracy(), Scorer)

    def test_numeric_index_prediction_matches(self):
        s = multiple_choice_accuracy()
        sample = _sample(["red", "green", "blue"])
        assert s.fitness(2, 2, sample) == 1.0

    def test_numeric_index_prediction_mismatches(self):
        s = multiple_choice_accuracy()
        sample = _sample(["red", "green", "blue"])
        assert s.fitness(0, 2, sample) == 0.0

    def test_letter_label_prediction_matches(self):
        s = multiple_choice_accuracy()
        sample = _sample(["red", "green", "blue"])
        assert s.fitness("C", 2, sample) == 1.0

    def test_letter_label_prediction_is_case_insensitive(self):
        s = multiple_choice_accuracy()
        sample = _sample(["red", "green", "blue"])
        assert s.fitness("c", 2, sample) == 1.0

    def test_freeform_text_naming_the_choice_matches(self):
        s = multiple_choice_accuracy()
        sample = _sample(["red", "green", "blue"])
        assert s.fitness("I think the answer is blue", 2, sample) == 1.0

    def test_answer_marker_is_used_to_disambiguate(self):
        s = multiple_choice_accuracy()
        sample = _sample(["red", "green", "blue"])
        # "red" appears in the reasoning but the marked answer is "blue".
        assert s.fitness("red is a nice color but Answer: blue", 2, sample) == 1.0

    def test_unparseable_prediction_scores_zero_and_flags_invalid(self):
        s = multiple_choice_accuracy()
        sample = _sample(["red", "green", "blue"])
        score = s.evaluate("xyzzy nonsense", 2, sample)
        assert score.value == 0.0
        assert score.breakdown["valid"] is False

    def test_none_prediction_scores_zero(self):
        s = multiple_choice_accuracy()
        sample = _sample(["red", "green", "blue"])
        assert s.fitness(None, 2, sample) == 0.0

    def test_missing_sample_meta_does_not_crash(self):
        s = multiple_choice_accuracy()
        bare_sample = Sample(id="s1", input={})
        assert s.fitness("A", 0, bare_sample) == 0.0

    def test_none_sample_does_not_crash(self):
        s = multiple_choice_accuracy()
        assert s.fitness("A", 0, None) == 0.0

    def test_aggregate_excludes_invalid_predictions_from_accuracy(self):
        s = multiple_choice_accuracy()
        sample = _sample(["red", "green", "blue"])
        scores = [
            s.evaluate(2, 2, sample),  # valid, correct
            s.evaluate(0, 2, sample),  # valid, incorrect
            s.evaluate("garbage", 2, sample),  # invalid, excluded
        ]
        agg = s.aggregate(scores)
        assert agg["fitness"] == pytest.approx(0.5)

    def test_repr_round_trips(self):
        s = multiple_choice_accuracy(strict=False)
        assert parse_scorer(repr(s)) == s
