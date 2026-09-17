"""Tests for task type schemas and validation."""

import pytest

from src.tasks.common.task_config_schema import (
    TASK_TYPE_SCHEMAS,
    validate_task_config,
    apply_defaults,
    get_handler_class,
    build_task_metadata,
    list_task_types,
    get_required_fields,
    get_optional_fields,
    get_task_type_description,
)


# Schema Definitions Tests

def test_task_type_schemas_has_all_types():
    """Test TASK_TYPE_SCHEMAS contains all expected task types."""
    assert "classification" in TASK_TYPE_SCHEMAS
    assert "structured" in TASK_TYPE_SCHEMAS
    assert "freeform" in TASK_TYPE_SCHEMAS


def test_task_type_schemas_classification_structure():
    """Test classification schema has required structure."""
    schema = TASK_TYPE_SCHEMAS["classification"]
    
    assert "handler_module" in schema
    assert "handler_class" in schema
    assert "required_fields" in schema
    assert "default_field_mappings" in schema
    assert "default_metrics" in schema
    assert "description" in schema


def test_task_type_schemas_structured_structure():
    """Test structured schema has required structure."""
    schema = TASK_TYPE_SCHEMAS["structured"]
    
    assert "handler_module" in schema
    assert "handler_class" in schema
    assert "required_fields" in schema
    assert "requires_one_of" in schema  # Schema requirement


def test_task_type_schemas_freeform_structure():
    """Test freeform schema has required structure."""
    schema = TASK_TYPE_SCHEMAS["freeform"]
    
    assert "handler_module" in schema
    assert "handler_class" in schema
    assert "default_field_mappings" in schema


# Validation Tests

def test_validate_task_config_classification_valid():
    """Test validate_task_config with valid classification config."""
    config = {
        "task_type": "classification",
        "dataset": {
            "name": "test/dataset",
            "input_field": "text",
            "output_field": "label",
            "labels": {"0": "No", "1": "Yes"}
        }
    }
    
    errors = validate_task_config("classification", config)
    
    assert len(errors) == 0


def test_validate_task_config_classification_missing_labels():
    """Test validate_task_config detects missing labels."""
    config = {
        "task_type": "classification",
        "dataset": {
            "name": "test/dataset",
            "input_field": "text",
            "output_field": "label"
            # Missing labels or choices_field
        }
    }
    
    errors = validate_task_config("classification", config)
    
    # Should have error about missing labels
    assert len(errors) > 0


def test_validate_task_config_structured_valid():
    """Test validate_task_config with valid structured config."""
    config = {
        "task_type": "structured",
        "dataset": {
            "name": "test/dataset",
            "input_field": "text",
            "output_field": "extracted",
            "schema_path": "/path/to/schema.json"
        }
    }
    
    errors = validate_task_config("structured", config)
    
    assert len(errors) == 0


def test_validate_task_config_structured_missing_schema():
    """Test validate_task_config detects missing schema."""
    config = {
        "task_type": "structured",
        "dataset": {
            "name": "test/dataset",
            "input_field": "text",
            "output_field": "extracted"
            # Missing schema_field, schema_path, or schema_json
        }
    }
    
    errors = validate_task_config("structured", config)
    
    # Should have error about missing schema
    assert len(errors) > 0


def test_validate_task_config_freeform_valid():
    """Test validate_task_config with valid freeform config."""
    config = {
        "task_type": "freeform",
        "dataset": {
            "name": "test/dataset",
            "input_field": "text",
            "output_field": "expected"
        }
    }
    
    errors = validate_task_config("freeform", config)
    
    assert len(errors) == 0


def test_validate_task_config_invalid_task_type():
    """Test validate_task_config with invalid task type."""
    config = {
        "task_type": "invalid_type",
        "dataset": {"name": "test/dataset"}
    }
    
    errors = validate_task_config("invalid_type", config)
    
    assert len(errors) > 0
    assert any("invalid" in err.lower() for err in errors)


def test_validate_task_config_missing_required_fields():
    """Test validate_task_config detects missing required fields."""
    config = {
        "task_type": "classification",
        "dataset": {
            "name": "test/dataset"
            # Missing input_field and output_field
        }
    }
    
    errors = validate_task_config("classification", config)
    
    # Should have errors about missing fields
    assert len(errors) > 0


# Defaults Application Tests

def test_apply_defaults_classification():
    """Test apply_defaults for classification task."""
    config = {
        "task_type": "classification",
        "dataset": {
            "name": "test/dataset",
            "labels": {"0": "No", "1": "Yes"}
        }
    }
    
    result = apply_defaults("classification", config)
    
    # Should have default field mappings
    assert "input_field" in result["dataset"]
    assert "output_field" in result["dataset"]


def test_apply_defaults_structured():
    """Test apply_defaults for structured task."""
    config = {
        "task_type": "structured",
        "dataset": {
            "name": "test/dataset",
            "schema_path": "/path/to/schema.json"
        }
    }
    
    result = apply_defaults("structured", config)
    
    # Should have default field mappings
    assert "input_field" in result["dataset"]
    assert "output_field" in result["dataset"]


def test_apply_defaults_freeform():
    """Test apply_defaults for freeform task."""
    config = {
        "task_type": "freeform",
        "dataset": {
            "name": "test/dataset"
        }
    }
    
    result = apply_defaults("freeform", config)
    
    # Should have default field mappings
    assert "input_field" in result["dataset"]
    assert "output_field" in result["dataset"]


def test_apply_defaults_sets_answer_type():
    """Test apply_defaults sets answer_type."""
    config = {
        "task_type": "classification",
        "dataset": {
            "name": "test/dataset",
            "labels": {"0": "No", "1": "Yes"}
        }
    }
    
    result = apply_defaults("classification", config)
    
    # Should have answer_type set
    assert "answer_type" in result


# Utilities Tests

def test_get_handler_class_classification():
    """Test get_handler_class for classification."""
    handler_class = get_handler_class("classification")
    
    assert handler_class is not None
    assert handler_class.__name__ == "MultipleChoiceHandler"


def test_get_handler_class_structured():
    """Test get_handler_class for structured."""
    handler_class = get_handler_class("structured")
    
    assert handler_class is not None
    assert handler_class.__name__ == "StructuredHandler"


def test_get_handler_class_freeform():
    """Test get_handler_class for freeform."""
    handler_class = get_handler_class("freeform")
    
    assert handler_class is not None
    assert handler_class.__name__ == "FreeformHandler"


def test_get_handler_class_invalid_raises():
    """Test get_handler_class raises for invalid task type."""
    with pytest.raises(ValueError, match="Invalid task"):
        get_handler_class("invalid_type")


def test_build_task_metadata():
    """Test build_task_metadata creates metadata dict."""
    config = {
        "task_type": "classification",
        "dataset": {
            "name": "test/dataset",
            "labels": {"0": "No", "1": "Yes"}
        }
    }
    
    metadata = build_task_metadata("classification", config)
    
    assert "display_name" in metadata
    assert "description" in metadata
    assert "capability_requirements" in metadata


def test_list_task_types():
    """Test list_task_types returns all task types."""
    task_types = list_task_types()
    
    assert "classification" in task_types
    assert "structured" in task_types
    assert "freeform" in task_types


def test_get_required_fields():
    """Test get_required_fields returns required fields."""
    fields = get_required_fields("classification")
    
    assert "input_field" in fields
    assert "output_field" in fields


def test_get_optional_fields():
    """Test get_optional_fields returns optional fields."""
    fields = get_optional_fields("classification")
    
    # Should have optional fields like system_prompt, etc.
    assert isinstance(fields, list)


def test_get_task_type_description():
    """Test get_task_type_description returns description."""
    description = get_task_type_description("classification")
    
    assert isinstance(description, str)
    assert len(description) > 0
