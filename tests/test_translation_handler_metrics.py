"""Tests for TranslationHandler aggregate metric metadata."""

from src.tasks.translation import metrics as translation_metrics
from src.tasks.translation._translation_handler import TranslationHandler


def test_translation_handler_aggregate_surfaces_metric_warnings(monkeypatch):
    monkeypatch.setattr(translation_metrics, "SACREBLEU_AVAILABLE", False)
    monkeypatch.setattr(translation_metrics, "COMET_AVAILABLE", False)

    handler = TranslationHandler(config={})
    sample_metrics = handler.calculate_metrics(
        prediction="hola mundo",
        expected="hola mundo",
        sample={"language_pair": "eng_spa", "direction": "eng->spa"},
    )
    aggregated = handler.aggregate_metrics([sample_metrics])

    assert aggregated["metric_degraded"] is True
    assert "sacrebleu_not_available" in aggregated["metric_warnings"]
    assert "comet_not_available" in aggregated["metric_warnings"]
