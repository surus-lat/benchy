"""Tests for TranscriptionHandler base."""

import pytest

from src.tasks.transcription._transcription_handler import TranscriptionHandler

pytest.importorskip("jiwer", reason="jiwer is required for WER/CER computation")


def _sample(locale: str = "es_419") -> dict:
    return {"id": "x", "expected": "hola mundo", "locale": locale}


def test_calculate_metrics_valid_sample_populates_all_metrics():
    h = TranscriptionHandler()
    metrics = h.calculate_metrics("hola mundo", "hola mundo", _sample())
    assert metrics["valid"] is True
    assert metrics["wer"] == 0.0
    assert metrics["cer"] == 0.0
    assert metrics["exact_match"] == 1.0
    assert metrics["locale"] == "es_419"


def test_calculate_metrics_with_error_marks_invalid():
    h = TranscriptionHandler()
    metrics = h.calculate_metrics(None, "hola", _sample(), error="boom", error_type="connectivity_error")
    assert metrics["valid"] is False
    assert metrics["error"] == "boom"
    assert metrics["error_type"] == "connectivity_error"
    assert metrics["locale"] == "es_419"


def test_calculate_metrics_with_none_prediction_marks_invalid():
    h = TranscriptionHandler()
    metrics = h.calculate_metrics(None, "hola", _sample())
    assert metrics["valid"] is False


def test_normalize_transcription_lowercases_and_collapses_whitespace():
    h = TranscriptionHandler()
    assert h._normalize_transcription("  HOLA   Mundo\n") == "hola mundo"


def test_aggregate_metrics_empty_list_returns_zeros():
    h = TranscriptionHandler()
    out = h.aggregate_metrics([])
    assert out["total_samples"] == 0
    assert out["valid_samples"] == 0
    assert out["per_locale"] == {}


def test_aggregate_metrics_buckets_by_locale():
    h = TranscriptionHandler()
    samples = [
        h.calculate_metrics("hola mundo", "hola mundo", _sample("es_419")),
        h.calculate_metrics("ola mundo", "hola mundo", _sample("es_419")),
        h.calculate_metrics("ola mundo", "ola mundo", _sample("pt_br")),
    ]
    out = h.aggregate_metrics(samples)
    assert set(out["per_locale"].keys()) == {"es_419", "pt_br"}
    assert out["per_locale"]["pt_br"]["wer"] == 0.0
    assert out["per_locale"]["es_419"]["sample_count"] == 2
    assert out["per_locale"]["pt_br"]["sample_count"] == 1


def test_aggregate_metrics_preinits_keys_when_all_failed():
    h = TranscriptionHandler()
    samples = [
        h.calculate_metrics(None, "hola", _sample(), error="boom"),
        h.calculate_metrics(None, "hola", _sample(), error="boom"),
    ]
    out = h.aggregate_metrics(samples)
    assert "wer" in out and out["wer"] is None
    assert "cer" in out and out["cer"] is None
    assert "exact_match" in out and out["exact_match"] is None
    assert out["valid_samples"] == 0
    assert out["total_samples"] == 2


def test_aggregate_metrics_error_rate_rounds():
    h = TranscriptionHandler()
    samples = [
        h.calculate_metrics("hola mundo", "hola mundo", _sample()),
        h.calculate_metrics(None, "hola", _sample(), error="boom"),
        h.calculate_metrics(None, "hola", _sample(), error="boom"),
    ]
    out = h.aggregate_metrics(samples)
    assert out["error_rate"] == round(2 / 3, 4)
    assert out["valid_samples"] == 1
    assert out["total_samples"] == 3
