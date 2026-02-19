"""Tests for StructuredHandler extraction logic."""

import json

import pytest

from src.tasks.common.structured import StructuredHandler


class SimpleStructuredHandler(StructuredHandler):
    """Simple structured handler for testing."""
    
    dataset_name = "test/dataset"
    text_field = "text"
    label_field = "expected"
    schema_field = "schema"


# Initialization & Schema Loading Tests

def test_init_with_schema_field_config():
    """Test initialization with schema_field in config."""
    config = {
        "dataset": {
            "schema_field": "custom_schema"
        }
    }
    
    handler = SimpleStructuredHandler(config)
    
    assert handler.schema_field == "custom_schema"


def test_init_with_schema_path_config(tmp_path):
    """Test initialization with schema_path in config."""
    schema_file = tmp_path / "schema.json"
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    
    with open(schema_file, 'w') as f:
        json.dump(schema, f)
    
    config = {
        "dataset": {
            "schema_path": str(schema_file)
        }
    }
    
    handler = SimpleStructuredHandler(config)
    
    assert hasattr(handler, '_global_schema')
    assert handler._global_schema == schema


def test_init_with_schema_json_config():
    """Test initialization with inline schema_json in config."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    
    config = {
        "dataset": {
            "schema_json": json.dumps(schema)
        }
    }
    
    handler = SimpleStructuredHandler(config)
    
    assert hasattr(handler, '_global_schema')
    assert handler._global_schema == schema


def test_init_with_missing_schema_file_raises(tmp_path):
    """Test initialization with non-existent schema_path raises clearly."""
    config = {
        "dataset": {
            "schema_path": str(tmp_path / "nonexistent.json")
        }
    }
    
    with pytest.raises(FileNotFoundError, match="Schema file not found"):
        SimpleStructuredHandler(config)


def test_apply_config_overrides_field_mappings():
    """Test _apply_config_overrides updates field mappings."""
    config = {
        "dataset": {
            "input_field": "document",
            "output_field": "extracted",
            "schema_field": "json_schema",
        }
    }
    
    handler = SimpleStructuredHandler(config)
    
    assert handler.text_field == "document"
    assert handler.label_field == "extracted"
    assert handler.schema_field == "json_schema"


def test_apply_config_overrides_schema_from_path(tmp_path):
    """Test _apply_config_overrides loads schema from path."""
    schema_file = tmp_path / "test_schema.json"
    schema = {"type": "object", "properties": {"field": {"type": "string"}}}
    
    with open(schema_file, 'w') as f:
        json.dump(schema, f)
    
    config = {
        "dataset": {
            "schema_path": str(schema_file)
        }
    }
    
    handler = SimpleStructuredHandler(config)
    
    assert handler._global_schema == schema


def test_apply_config_overrides_schema_from_json():
    """Test _apply_config_overrides parses inline schema JSON."""
    schema = {"type": "object", "properties": {"field": {"type": "string"}}}
    
    config = {
        "dataset": {
            "schema_json": json.dumps(schema)
        }
    }
    
    handler = SimpleStructuredHandler(config)
    
    assert handler._global_schema == schema


# Sample Preprocessing Tests

def test_preprocess_sample_with_schema_in_sample():
    """Test preprocessing when schema is in the sample."""
    handler = SimpleStructuredHandler()
    
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    raw_sample = {
        "text": "Extract name from this",
        "schema": schema,
        "expected": {"name": "John"}
    }
    
    result = handler.preprocess_sample(raw_sample, 0)
    
    assert result is not None
    assert result["schema"] == schema
    assert result["expected"] == {"name": "John"}


def test_preprocess_sample_with_global_schema(tmp_path):
    """Test preprocessing when schema is provided globally via config."""
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    
    config = {
        "dataset": {
            "schema_json": json.dumps(schema)
        }
    }
    
    handler = SimpleStructuredHandler(config)
    
    raw_sample = {
        "text": "Extract name from this",
        "expected": {"name": "John"}
    }
    
    result = handler.preprocess_sample(raw_sample, 0)
    
    assert result is not None
    assert result["schema"] == schema


def test_preprocess_sample_already_preprocessed_passes_through():
    """Test preprocessing passes through already preprocessed samples."""
    handler = SimpleStructuredHandler()
    
    schema = {"type": "object"}
    preprocessed = {
        "_preprocessed": True,
        "id": "test_1",
        "text": "Sample",
        "schema": schema,
        "expected": {"field": "value"}
    }
    
    result = handler.preprocess_sample(preprocessed, 0)
    
    assert result == preprocessed


def test_preprocess_sample_skips_on_missing_text():
    """Test preprocessing skips sample when text field missing."""
    handler = SimpleStructuredHandler()
    
    raw_sample = {
        "schema": {"type": "object"},
        "expected": {"field": "value"}
    }
    
    result = handler.preprocess_sample(raw_sample, 0)
    
    assert result is None


def test_preprocess_sample_skips_on_missing_expected():
    """Test preprocessing skips sample when expected field missing."""
    handler = SimpleStructuredHandler()
    
    raw_sample = {
        "text": "Sample text",
        "schema": {"type": "object"}
    }
    
    result = handler.preprocess_sample(raw_sample, 0)
    
    assert result is None


def test_preprocess_sample_skips_on_missing_schema():
    """Test preprocessing skips sample when schema missing."""
    handler = SimpleStructuredHandler()
    # No global schema, no schema in sample
    
    raw_sample = {
        "text": "Sample text",
        "expected": {"field": "value"}
    }
    
    result = handler.preprocess_sample(raw_sample, 0)
    
    assert result is None


# Metrics Calculator Tests

def test_metrics_calculator_lazy_initialization():
    """Test metrics_calculator property is lazily initialized."""
    handler = SimpleStructuredHandler()
    
    # Should not be initialized yet
    assert handler._metrics_calc is None
    
    # Access property
    calc = handler.metrics_calculator
    
    # Should now be initialized
    assert calc is not None
    assert handler._metrics_calc is not None


def test_metrics_calculator_merges_config():
    """Test metrics_calculator merges config with defaults."""
    handler = SimpleStructuredHandler()
    handler.metrics_config = {"strict": False}
    handler.config = {"metrics": {"strict": True}}
    
    calc = handler.metrics_calculator
    
    # Config should override defaults
    assert calc is not None


def test_build_additional_artifacts_writes_field_diagnostics(tmp_path):
    """Structured handler writes field diagnostics artifacts from per-sample metrics."""
    handler = SimpleStructuredHandler()
    results = {
        "per_sample_metrics": [
            {
                "sample_id": "s1",
                "schema_fingerprint": "schema_1",
                "field_results": {
                    "total": {
                        "match_type": "incorrect",
                        "type_match": False,
                        "expected": 100,
                        "predicted": "100.00",
                    }
                },
            },
            {
                "sample_id": "s2",
                "schema_fingerprint": "schema_1",
                "field_results": {
                    "total": {
                        "match_type": "exact",
                        "type_match": True,
                        "expected": 200,
                        "predicted": 200,
                    }
                },
            },
        ]
    }

    artifacts = handler.build_additional_artifacts(
        results=results,
        output_dir=tmp_path,
        safe_model_name="model_x",
        timestamp="20260218_180000",
        task_name="simple_structured",
    )

    assert len(artifacts) == 2
    assert all(path.exists() for path in artifacts)
    assert artifacts[0].name.endswith("_field_diagnostics.json")
    assert artifacts[1].name.endswith("_field_diagnostics.txt")


def test_build_additional_artifacts_skips_multiple_schema_fingerprints(tmp_path):
    """Structured handler skips diagnostics when run has mixed schemas."""
    handler = SimpleStructuredHandler()
    results = {
        "per_sample_metrics": [
            {"sample_id": "s1", "schema_fingerprint": "schema_a", "field_results": {"a": {"match_type": "exact"}}},
            {"sample_id": "s2", "schema_fingerprint": "schema_b", "field_results": {"a": {"match_type": "exact"}}},
        ]
    }

    artifacts = handler.build_additional_artifacts(
        results=results,
        output_dir=tmp_path,
        safe_model_name="model_x",
        timestamp="20260218_180001",
        task_name="simple_structured",
    )

    assert artifacts == []
