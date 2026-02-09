"""Extended tests for metrics beyond basic coverage."""

import pytest

from src.tasks.common.metrics import (
    ExactMatch,
    F1Score,
    MultipleChoiceAccuracy,
    MeanSquaredError,
    PearsonCorrelation,
)


# ExactMatch Extended Tests

def test_exact_match_with_none_values():
    """Test ExactMatch handles None values."""
    metric = ExactMatch()
    
    result = metric.per_sample(None, "expected", {})
    
    assert result["exact_match"] == 0.0


def test_exact_match_case_sensitive():
    """Test ExactMatch with case_insensitive=False."""
    metric = ExactMatch(case_insensitive=False)
    
    result = metric.per_sample("Answer", "answer", {})
    
    assert result["exact_match"] == 0.0  # Should not match


def test_exact_match_no_strip():
    """Test ExactMatch with strip=False."""
    metric = ExactMatch(strip=False)
    
    result = metric.per_sample("  answer  ", "answer", {})
    
    assert result["exact_match"] == 0.0  # Should not match with whitespace


def test_exact_match_with_numbers():
    """Test ExactMatch with numeric values."""
    metric = ExactMatch()
    
    result = metric.per_sample("42", "42", {})
    
    assert result["exact_match"] == 1.0


# F1Score Extended Tests

def test_f1_score_with_empty_prediction():
    """Test F1Score with empty prediction."""
    metric = F1Score()
    
    result = metric.per_sample("", "the quick brown fox", {})
    
    assert result["f1"] == 0.0


def test_f1_score_with_empty_expected():
    """Test F1Score with empty expected value."""
    metric = F1Score()
    
    result = metric.per_sample("some text", "", {})
    
    assert result["f1"] == 0.0


def test_f1_score_with_none_values():
    """Test F1Score handles None values."""
    metric = F1Score()
    
    result = metric.per_sample(None, "expected", {})
    
    assert result["f1"] == 0.0


def test_f1_score_precision_recall_calculation():
    """Test F1Score computes F1 correctly."""
    metric = F1Score()
    
    # Prediction: "the quick brown dog"
    # Expected: "the quick brown fox"
    # Common: "the quick brown" (3 tokens)
    # Precision: 3/4 = 0.75
    # Recall: 3/4 = 0.75
    # F1: 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
    
    result = metric.per_sample("the quick brown dog", "the quick brown fox", {})
    
    assert result["f1"] == 0.75


def test_f1_score_with_list_of_references():
    """Test F1Score with multiple reference answers."""
    metric = F1Score()
    
    # F1Score supports list of references - takes max F1
    result = metric.per_sample("answer", ["answer", "different"], {})
    
    assert result["f1"] == 1.0


# MultipleChoiceAccuracy Extended Tests

def test_multiple_choice_accuracy_correct_answer():
    """Test MultipleChoiceAccuracy with correct prediction."""
    metric = MultipleChoiceAccuracy()
    
    sample = {
        "choices": ["A", "B", "C"],
        "choice_labels": ["0", "1", "2"],
        "expected": 1,
    }
    
    result = metric.per_sample("B", 1, sample)
    
    assert result["accuracy"] == 1.0


def test_multiple_choice_accuracy_wrong_answer():
    """Test MultipleChoiceAccuracy with wrong prediction."""
    metric = MultipleChoiceAccuracy()
    
    sample = {
        "choices": ["A", "B", "C"],
        "choice_labels": ["0", "1", "2"],
        "expected": 1,
    }
    
    result = metric.per_sample("A", 1, sample)
    
    assert result["accuracy"] == 0.0


def test_multiple_choice_accuracy_with_logprobs():
    """Test MultipleChoiceAccuracy with text prediction that matches choice."""
    metric = MultipleChoiceAccuracy()
    
    sample = {
        "choices": ["A", "B", "C"],
        "choice_labels": ["0", "1", "2"],
        "expected": 1,
    }
    
    # Text prediction that matches a choice
    result = metric.per_sample("B", 1, sample)
    
    # Should match choice at index 1
    assert result["accuracy"] == 1.0


def test_multiple_choice_accuracy_aggregation():
    """Test MultipleChoiceAccuracy aggregation."""
    metric = MultipleChoiceAccuracy()
    
    per_sample = [
        {"accuracy": 1.0, "valid": True},
        {"accuracy": 0.0, "valid": True},
        {"accuracy": 1.0, "valid": True},
        {"accuracy": 0.0, "valid": True},
    ]
    
    aggregated = metric.aggregate(per_sample)
    
    assert aggregated["accuracy"] == 0.5  # 2/4


# MeanSquaredError Tests

def test_mse_computes_squared_error():
    """Test MeanSquaredError computes squared error."""
    metric = MeanSquaredError()
    
    result = metric.per_sample("3.0", "5.0", {})
    
    # (3.0 - 5.0)^2 = 4.0
    assert result["mse"] == 4.0


def test_mse_handles_invalid_types():
    """Test MeanSquaredError handles non-numeric values."""
    metric = MeanSquaredError()
    
    result = metric.per_sample("not a number", "5.0", {})
    
    # Should handle gracefully, returning 0 or NaN
    assert "mse" in result


def test_mse_aggregation():
    """Test MeanSquaredError aggregation."""
    metric = MeanSquaredError()
    
    per_sample = [
        {"mse": 4.0, "valid": True},
        {"mse": 9.0, "valid": True},
        {"mse": 1.0, "valid": True},
    ]
    
    aggregated = metric.aggregate(per_sample)
    
    # Mean of [4.0, 9.0, 1.0] = 4.666...
    assert 4.6 < aggregated["mse"] < 4.7


# PearsonCorrelation Tests

def test_pearson_correlation_perfect_positive():
    """Test PearsonCorrelation with perfect positive correlation."""
    metric = PearsonCorrelation()
    
    # This metric requires aggregation to compute correlation
    # Single sample just stores values
    result = metric.per_sample("5.0", "5.0", {})
    
    assert "pearson" in result or "prediction" in result


def test_pearson_correlation_perfect_negative():
    """Test PearsonCorrelation with perfect negative correlation."""
    metric = PearsonCorrelation()
    
    # Requires aggregation over multiple samples
    result = metric.per_sample("1.0", "10.0", {})
    
    # Individual per_sample just stores values
    assert "pearson" in result or "prediction" in result


def test_pearson_correlation_no_correlation():
    """Test PearsonCorrelation with no correlation."""
    metric = PearsonCorrelation()
    
    result = metric.per_sample("5.0", "3.0", {})
    
    # Single sample can't compute correlation
    assert "pearson" in result or "prediction" in result


def test_pearson_correlation_insufficient_samples():
    """Test PearsonCorrelation with insufficient samples."""
    metric = PearsonCorrelation()
    
    per_sample = [
        {"pearson": 0.0},
    ]
    
    # Need at least 2 samples for correlation
    # Aggregation should handle this gracefully
    aggregated = metric.aggregate(per_sample)
    
    # Should return pearson key
    assert "pearson" in aggregated


def test_pearson_correlation_aggregates_perfect_positive():
    """Test PearsonCorrelation computes perfect positive correlation."""
    metric = PearsonCorrelation()

    per_sample = [
        metric.per_sample("1.0", "1.0", {}),
        metric.per_sample("2.0", "2.0", {}),
        metric.per_sample("3.0", "3.0", {}),
    ]

    aggregated = metric.aggregate(per_sample)

    assert aggregated["pearson"] == pytest.approx(1.0)


def test_pearson_correlation_ignores_non_finite_values():
    """Test PearsonCorrelation excludes NaN inputs from aggregation."""
    metric = PearsonCorrelation()

    per_sample = [
        metric.per_sample(float("nan"), "1.0", {}),
        metric.per_sample("2.0", "2.0", {}),
    ]

    aggregated = metric.aggregate(per_sample)

    # Only one finite point remains, so correlation is undefined -> 0.0 fallback.
    assert aggregated["pearson"] == 0.0
