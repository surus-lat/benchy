"""Tests for translation metric computation and aggregation."""

from dataclasses import dataclass

from src.tasks.translation import metrics as translation_metrics


@dataclass
class _Score:
    score: float


class _FakeSacrebleu:
    def sentence_bleu(self, prediction, refs):
        if prediction == "p1":
            return _Score(10.0)
        return _Score(90.0)

    def sentence_chrf(self, prediction, refs):
        if prediction == "p1":
            return _Score(20.0)
        return _Score(80.0)

    def corpus_bleu(self, predictions, refs):
        return _Score(55.0)

    def corpus_chrf(self, predictions, refs):
        return _Score(65.0)


def test_aggregate_uses_corpus_bleu_and_chrf(monkeypatch):
    monkeypatch.setattr(translation_metrics, "SACREBLEU_AVAILABLE", True)
    monkeypatch.setattr(translation_metrics, "COMET_AVAILABLE", False)
    monkeypatch.setattr(translation_metrics, "sacrebleu", _FakeSacrebleu(), raising=False)

    calc = translation_metrics.TranslationMetricsCalculator({})
    m1 = calc.calculate("p1", "r1")
    m2 = calc.calculate("p2", "r2")
    aggregated = calc.aggregate([m1, m2])

    assert m1["bleu"] == 0.1
    assert m2["bleu"] == 0.9
    assert aggregated["bleu"] == 0.55
    assert aggregated["chrf"] == 0.65
    assert aggregated["bleu_aggregation"] == "corpus"
    assert aggregated["chrf_aggregation"] == "corpus"


def test_calculate_marks_degraded_when_metric_dependencies_missing(monkeypatch):
    monkeypatch.setattr(translation_metrics, "SACREBLEU_AVAILABLE", False)
    monkeypatch.setattr(translation_metrics, "COMET_AVAILABLE", False)

    calc = translation_metrics.TranslationMetricsCalculator({})
    sample_metrics = calc.calculate("hola mundo", "hola mundo")
    aggregated = calc.aggregate([sample_metrics])

    assert sample_metrics["valid"] is True
    assert sample_metrics["metric_degraded"] is True
    assert "sacrebleu_not_available" in sample_metrics["metric_warnings"]
    assert "comet_not_available" in sample_metrics["metric_warnings"]
    assert aggregated["metric_degraded"] is True
    assert "sacrebleu_not_available" in aggregated["metric_warnings"]
    assert "comet_not_available" in aggregated["metric_warnings"]
