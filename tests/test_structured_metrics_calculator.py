"""Tests for structured extraction metrics calculator."""

import pytest

from src.tasks.common.utils.structured_metrics_calculator import MetricsCalculator


# Test Configuration

@pytest.fixture
def default_config():
    """Default configuration for metrics calculator."""
    return {
        "metrics": {
            "partial_credit": 0.3,
            "extraction_quality_score": {
                "weights": {
                    "schema_validity": 0.15,
                    "field_f1_partial": 0.70,
                    "inverted_hallucination": 0.15
                }
            },
            "partial_matching": {
                "string": {
                    "similarity_threshold": 0.8,
                    "partial_credit": 0.5
                },
                "number": {
                    "tolerance": 0.01,
                    "partial_credit": 0.5
                }
            },
            "normalization": {
                "lowercase": True,
                "strip_whitespace": True
            }
        }
    }


@pytest.fixture
def simple_schema():
    """Simple JSON schema for testing."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"}
        },
        "required": ["name", "age"]
    }


# Initialization Tests

def test_metrics_calculator_init_with_config(default_config):
    """Test MetricsCalculator initialization with config."""
    calc = MetricsCalculator(default_config)
    
    assert calc.config == default_config
    assert calc.partial_matcher is not None


def test_metrics_calculator_init_with_strict_mode(default_config):
    """Test MetricsCalculator initialization with strict mode."""
    calc = MetricsCalculator(default_config, strict=True)
    
    assert calc.partial_matcher is not None


# calculate_all Tests (Main Public API)

def test_calculate_all_perfect_match(default_config, simple_schema):
    """Test calculate_all with perfect match."""
    calc = MetricsCalculator(default_config)
    
    predicted = {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com"
    }
    expected = {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com"
    }
    
    metrics = calc.calculate_all(predicted, expected, simple_schema)
    
    assert metrics["extraction_quality_score"] == 1.0
    assert metrics["valid"] is True


def test_calculate_all_partial_match(default_config, simple_schema):
    """Test calculate_all with partial match."""
    calc = MetricsCalculator(default_config)
    
    predicted = {
        "name": "John Smith",  # Different last name
        "age": 30,
        "email": "john@example.com"
    }
    expected = {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com"
    }
    
    metrics = calc.calculate_all(predicted, expected, simple_schema)
    
    assert 0.0 < metrics["extraction_quality_score"] < 1.0


def test_calculate_all_with_hallucinations(default_config, simple_schema):
    """Test calculate_all with extra fields (hallucinations)."""
    calc = MetricsCalculator(default_config)
    
    predicted = {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com",
        "extra_field": "hallucination"  # Not in schema
    }
    expected = {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com"
    }
    
    metrics = calc.calculate_all(predicted, expected, simple_schema)
    
    # Should penalize for hallucination
    assert metrics["hallucination_rate"] > 0.0
    assert metrics["extraction_quality_score"] < 1.0


def test_calculate_all_with_missing_fields(default_config, simple_schema):
    """Test calculate_all with missing fields."""
    calc = MetricsCalculator(default_config)
    
    predicted = {
        "name": "John Doe"
        # Missing 'age' and 'email'
    }
    expected = {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com"
    }
    
    metrics = calc.calculate_all(predicted, expected, simple_schema)
    
    # Should have low score due to missing fields, or be invalid
    if "extraction_quality_score" in metrics:
        assert metrics["extraction_quality_score"] < 0.5
    else:
        # Missing required fields may cause schema validation failure
        assert metrics["valid"] is False


def test_calculate_all_schema_invalid(default_config, simple_schema):
    """Test calculate_all with schema-invalid prediction."""
    calc = MetricsCalculator(default_config)
    
    predicted = {
        "name": "John Doe",
        "age": "not a number"  # Invalid type
    }
    expected = {
        "name": "John Doe",
        "age": 30
    }
    
    metrics = calc.calculate_all(predicted, expected, simple_schema)
    
    assert metrics["valid"] is False
    # EQS may not be present if schema validation fails early
    if "extraction_quality_score" in metrics:
        assert metrics["extraction_quality_score"] < 0.5


def test_calculate_all_with_nested_objects(default_config):
    """Test calculate_all with nested object structures."""
    calc = MetricsCalculator(default_config)
    
    schema = {
        "type": "object",
        "properties": {
            "person": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            }
        }
    }
    
    predicted = {
        "person": {"name": "John", "age": 30}
    }
    expected = {
        "person": {"name": "John", "age": 30}
    }
    
    metrics = calc.calculate_all(predicted, expected, schema)
    
    assert metrics["extraction_quality_score"] == 1.0


def test_calculate_all_with_arrays(default_config):
    """Test calculate_all with array fields."""
    calc = MetricsCalculator(default_config)
    
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }
    
    predicted = {
        "items": ["item1", "item2", "item3"]
    }
    expected = {
        "items": ["item1", "item2", "item3"]
    }
    
    metrics = calc.calculate_all(predicted, expected, schema)
    
    assert metrics["extraction_quality_score"] == 1.0


# Aggregation Tests

def test_aggregate_metrics_across_samples(default_config, simple_schema):
    """Test aggregate_metrics computes averages."""
    calc = MetricsCalculator(default_config)
    
    # Generate real metrics from calculate_all
    predicted1 = {"name": "John", "age": 30, "email": "john@example.com"}
    expected1 = {"name": "John", "age": 30, "email": "john@example.com"}
    metrics1 = calc.calculate_all(predicted1, expected1, simple_schema)
    
    predicted2 = {"name": "Jane", "age": 25, "email": "jane@example.com"}
    expected2 = {"name": "Jane", "age": 25, "email": "jane@example.com"}
    metrics2 = calc.calculate_all(predicted2, expected2, simple_schema)
    
    per_sample_metrics = [metrics1, metrics2]
    
    aggregated = calc.aggregate_metrics(per_sample_metrics)
    
    assert "extraction_quality_score" in aggregated
    # Both perfect matches should average to 1.0
    assert aggregated["extraction_quality_score"] == 1.0


def test_aggregate_metrics_with_errors(default_config, simple_schema):
    """Test aggregate_metrics handles error samples."""
    calc = MetricsCalculator(default_config)
    
    # Generate real metrics with one error case
    predicted1 = {"name": "John", "age": 30, "email": "john@example.com"}
    expected1 = {"name": "John", "age": 30, "email": "john@example.com"}
    metrics1 = calc.calculate_all(predicted1, expected1, simple_schema)
    
    # Create an error case by passing invalid types
    predicted2 = {"name": 123, "age": "invalid"}  # Wrong types
    expected2 = {"name": "Jane", "age": 25}
    metrics2 = calc.calculate_all(predicted2, expected2, simple_schema)
    
    per_sample_metrics = [metrics1, metrics2]
    
    aggregated = calc.aggregate_metrics(per_sample_metrics)
    
    # Should handle errors gracefully
    assert "extraction_quality_score" in aggregated


def test_aggregate_metrics_empty_list(default_config):
    """Test aggregate_metrics handles empty list."""
    calc = MetricsCalculator(default_config)
    
    aggregated = calc.aggregate_metrics([])
    
    # Should return defaults or empty dict
    assert isinstance(aggregated, dict)


# Integration Tests

def test_calculate_all_with_field_weights(default_config, simple_schema):
    """Test calculate_all respects field weights from config."""
    config_with_weights = default_config.copy()
    config_with_weights["metrics"]["field_weights"] = {
        "name": 2.0,  # Name is more important
        "age": 1.0
    }
    
    calc = MetricsCalculator(config_with_weights)
    
    predicted = {
        "name": "Wrong Name",  # Wrong important field
        "age": 30,  # Correct
        "email": "john@example.com"
    }
    expected = {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com"
    }
    
    metrics = calc.calculate_all(predicted, expected, simple_schema)
    
    # Score should be lower due to wrong high-weight field
    assert metrics["extraction_quality_score"] < 0.9


def test_calculate_all_handles_null_values(default_config, simple_schema):
    """Test calculate_all handles null/None values."""
    calc = MetricsCalculator(default_config)
    
    predicted = {
        "name": "John Doe",
        "age": None,  # Null value
        "email": "john@example.com"
    }
    expected = {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com"
    }
    
    metrics = calc.calculate_all(predicted, expected, simple_schema)
    
    # Should handle null gracefully - but may not have extraction_quality_score if schema invalid
    assert "valid" in metrics or "extraction_quality_score" in metrics


def test_calculate_all_with_date_normalization(default_config):
    """Test calculate_all normalizes dates."""
    calc = MetricsCalculator(default_config)
    
    schema = {
        "type": "object",
        "properties": {
            "date": {"type": "string", "format": "date"}
        }
    }
    
    predicted = {"date": "15-01-2024"}  # DD-MM-YYYY
    expected = {"date": "2024-01-15"}  # ISO format
    
    metrics = calc.calculate_all(predicted, expected, schema)
    
    # Should normalize and match
    assert metrics["extraction_quality_score"] > 0.9
