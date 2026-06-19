"""Tests for WordErrorRate and CharErrorRate metrics."""

import math

import pytest

from src.tasks.common.metrics import CharErrorRate, WordErrorRate

pytest.importorskip("jiwer", reason="jiwer is required for WER/CER tests")


def test_wer_perfect_match_returns_zero():
    wer = WordErrorRate()
    assert wer.compute("hello world", "hello world", {}) == 0.0


def test_wer_complete_mismatch_high():
    wer = WordErrorRate()
    score = wer.compute("foo bar baz", "alpha beta gamma", {})
    assert score >= 1.0


def test_wer_empty_prediction_returns_one():
    wer = WordErrorRate()
    assert wer.compute("", "hello world", {}) == 1.0
    assert wer.compute(None, "hello world", {}) == 1.0


def test_cer_perfect_match_returns_zero():
    cer = CharErrorRate()
    assert cer.compute("hola", "hola", {}) == 0.0


def test_cer_partial_match_between_zero_and_one():
    cer = CharErrorRate()
    score = cer.compute("hola mundo", "hola mando", {})
    assert 0.0 < score < 1.0


def test_aggregate_averages_scores():
    wer = WordErrorRate()
    values = [{"wer": 0.0}, {"wer": 0.5}, {"wer": 1.0}]
    result = wer.aggregate(values)
    assert math.isclose(result["wer"], 0.5)


def test_aggregate_skips_nan_and_empty_returns_one():
    cer = CharErrorRate()
    assert cer.aggregate([])["cer"] == 1.0
    mixed = [{"cer": 0.2}, {"cer": float("nan")}, {"cer": 0.4}]
    avg = cer.aggregate(mixed)["cer"]
    assert math.isclose(avg, 0.3)
