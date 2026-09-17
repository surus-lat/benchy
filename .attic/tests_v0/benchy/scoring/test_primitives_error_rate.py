"""wer, cer — error-family transcription metrics.

Both are naturally "lower is better" in the literature (0.0 = perfect
transcription). `Score.value` / `fitness` invert that so higher is always
better, per the package-wide convention: `value = clamp(1 - raw_rate, 0, 1)`.
The raw (uninverted) rate is preserved in `breakdown` for diagnostics.
"""

from __future__ import annotations

import pytest

from benchy.core import Scorer
from benchy.scoring.primitives import cer, wer
from benchy.scoring.registry import parse_scorer


class TestWordErrorRate:
    def test_is_a_scorer(self):
        assert isinstance(wer(), Scorer)

    def test_identical_transcripts_score_perfect_fitness(self):
        s = wer()
        assert s.fitness("the cat sat on the mat", "the cat sat on the mat", None) == pytest.approx(1.0)

    def test_completely_wrong_transcript_scores_low_fitness(self):
        s = wer()
        value = s.fitness("completely unrelated text here", "the cat sat on the mat", None)
        assert value < 0.5

    def test_partial_errors_score_between_zero_and_one(self):
        s = wer()
        value = s.fitness("the cat sat on a mat", "the cat sat on the mat", None)
        assert 0.0 < value < 1.0

    def test_empty_prediction_scores_zero_fitness(self):
        s = wer()
        assert s.fitness("", "the cat sat", None) == 0.0

    def test_empty_expected_scores_zero_fitness(self):
        s = wer()
        assert s.fitness("the cat sat", "", None) == 0.0

    def test_none_prediction_scores_zero_fitness(self):
        s = wer()
        assert s.fitness(None, "the cat sat", None) == 0.0

    def test_breakdown_carries_the_raw_uninverted_wer(self):
        s = wer()
        score = s.evaluate("the cat sat on the mat", "the cat sat on the mat", None)
        assert score.breakdown["wer"] == pytest.approx(0.0)
        assert score.value == pytest.approx(1.0)

    def test_fitness_never_goes_negative_even_with_many_insertions(self):
        s = wer()
        # jiwer's wer can exceed 1.0 when there are more insertions than
        # reference words; fitness must still clamp into [0, 1].
        value = s.fitness("one two three four five six seven", "one", None)
        assert 0.0 <= value <= 1.0

    def test_aggregate_reports_fitness_and_raw_wer(self):
        s = wer()
        scores = [
            s.evaluate("a b c", "a b c", None),
            s.evaluate("x y z", "a b c", None),
        ]
        agg = s.aggregate(scores)
        assert "fitness" in agg
        assert "wer" in agg

    def test_repr_round_trips(self):
        s = wer()
        assert parse_scorer(repr(s)) == s


class TestCharErrorRate:
    def test_is_a_scorer(self):
        assert isinstance(cer(), Scorer)

    def test_identical_transcripts_score_perfect_fitness(self):
        s = cer()
        assert s.fitness("hello world", "hello world", None) == pytest.approx(1.0)

    def test_one_character_typo_scores_high_but_not_perfect(self):
        s = cer()
        value = s.fitness("hallo world", "hello world", None)
        assert 0.8 < value < 1.0

    def test_empty_prediction_scores_zero_fitness(self):
        s = cer()
        assert s.fitness("", "hello", None) == 0.0

    def test_none_expected_scores_zero_fitness(self):
        s = cer()
        assert s.fitness("hello", None, None) == 0.0

    def test_breakdown_carries_the_raw_uninverted_cer(self):
        s = cer()
        score = s.evaluate("hello world", "hello world", None)
        assert score.breakdown["cer"] == pytest.approx(0.0)

    def test_repr_round_trips(self):
        s = cer()
        assert parse_scorer(repr(s)) == s
