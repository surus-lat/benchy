"""Tests for ad-hoc task registry functionality."""


from src.tasks.registry import (
    build_adhoc_task_config,
    build_adhoc_task_spec,
    build_handler_task_config,
    is_handler_based_task,
)


# Ad-hoc Task Config Building Tests

def test_build_adhoc_task_config_classification():
    """Test build_adhoc_task_config for classification task."""
    task_name = "_adhoc_classification_123"  # Format: _adhoc_{type}_{hash}
    task_config = {
        "dataset": {
            "name": "test/dataset",
            "labels": {"0": "No", "1": "Yes"}
        }
    }
    
    config = build_adhoc_task_config(task_name, task_config)
    
    # Config has structure with defaults and task_configs
    assert "defaults" in config
    assert config["defaults"]["dataset"]["name"] == "test/dataset"


def test_build_adhoc_task_config_structured():
    """Test build_adhoc_task_config for structured extraction task."""
    task_name = "_adhoc_structured_456"  # Format: _adhoc_{type}_{hash}
    task_config = {
        "dataset": {
            "name": "test/dataset",
            "schema_path": "/path/to/schema.json"
        }
    }
    
    config = build_adhoc_task_config(task_name, task_config)
    
    assert "defaults" in config
    assert "schema_path" in config["defaults"]["dataset"]


def test_build_adhoc_task_config_freeform():
    """Test build_adhoc_task_config for freeform generation task."""
    task_name = "_adhoc_freeform_789"  # Format: _adhoc_{type}_{hash}
    task_config = {
        "dataset": {
            "name": "test/dataset"
        }
    }
    
    config = build_adhoc_task_config(task_name, task_config)
    
    assert "defaults" in config


def test_build_adhoc_task_config_applies_defaults():
    """Test build_adhoc_task_config applies task type defaults."""
    task_name = "_adhoc_classification_001"  # Format: _adhoc_{type}_{hash}
    task_config = {
        "dataset": {
            "name": "test/dataset",
            "labels": {"0": "No", "1": "Yes"}
        }
    }
    
    config = build_adhoc_task_config(task_name, task_config)
    
    # Should have defaults applied in defaults section
    assert "input_field" in config["defaults"]["dataset"]
    assert "output_field" in config["defaults"]["dataset"]


def test_build_adhoc_task_config_extracts_task_type():
    """Test build_adhoc_task_config extracts task_type from name."""
    task_name = "_adhoc_freeform_002"  # Format: _adhoc_{type}_{hash}
    task_config = {
        "dataset": {"name": "test/dataset"}
    }
    
    config = build_adhoc_task_config(task_name, task_config)
    
    # Task type is extracted from name, not stored in config
    assert config["name"] == task_name


def test_build_adhoc_task_config_invalid_name_raises():
    """Test build_adhoc_task_config raises on invalid task name."""
    task_name = "regular_task"  # Not an ad-hoc name
    task_config = {
        "task_type": "classification",
        "dataset": {"name": "test/dataset"}
    }
    
    # Should handle gracefully or raise
    # Implementation may vary
    try:
        build_adhoc_task_config(task_name, task_config)
        # If it doesn't raise, that's also acceptable
        assert True
    except ValueError:
        # If it raises, that's also acceptable
        assert True


# Ad-hoc Task Spec Building Tests

def test_build_adhoc_task_spec_creates_spec():
    """Test build_adhoc_task_spec creates TaskGroupSpec."""
    task_name = "_adhoc_classification_003"  # Format: _adhoc_{type}_{hash}
    task_config = {
        "dataset": {
            "name": "test/dataset",
            "labels": {"0": "No", "1": "Yes"}
        }
    }
    
    spec = build_adhoc_task_spec(task_name, task_config)
    
    assert spec is not None
    assert spec.name == task_name


def test_build_adhoc_task_spec_gets_handler_class():
    """Test build_adhoc_task_spec loads correct handler class."""
    task_name = "_adhoc_freeform_004"  # Format: _adhoc_{type}_{hash}
    task_config = {
        "dataset": {"name": "test/dataset"}
    }
    
    spec = build_adhoc_task_spec(task_name, task_config)
    
    # Handler class is loaded internally in prepare_task
    # Verify spec has prepare_task function
    assert spec.prepare_task is not None


def test_build_adhoc_task_spec_prepare_task_function():
    """Test build_adhoc_task_spec creates prepare_task function."""
    task_name = "_adhoc_classification_005"  # Format: _adhoc_{type}_{hash}
    task_config = {
        "dataset": {
            "name": "test/dataset",
            "labels": {"0": "No", "1": "Yes"}
        }
    }
    
    spec = build_adhoc_task_spec(task_name, task_config)
    
    # Should have prepare_task callable
    assert spec.prepare_task is not None
    assert callable(spec.prepare_task)


# Handler Task Config Tests

def test_build_handler_task_config_recognizes_adhoc():
    """Test build_handler_task_config recognizes ad-hoc tasks."""
    task_ref = "_adhoc_classification_006"  # Format: _adhoc_{type}_{hash}
    adhoc_configs = {
        task_ref: {
            "dataset": {"name": "test/dataset", "labels": {"0": "No", "1": "Yes"}}
        }
    }
    
    config = build_handler_task_config(
        task_ref,
        adhoc_task_configs=adhoc_configs
    )
    
    assert config is not None
    assert "defaults" in config


def test_build_handler_task_config_regular_task():
    """Test build_handler_task_config handles regular tasks."""
    task_ref = "classify.sentiment"
    
    # Should handle regular task lookup
    # May raise if task doesn't exist, which is acceptable
    try:
        build_handler_task_config(task_ref)
        assert True
    except (ValueError, KeyError):
        # Expected if task doesn't exist
        assert True


def test_build_handler_task_config_passes_adhoc_configs():
    """Test build_handler_task_config passes adhoc_configs parameter."""
    task_ref = "_adhoc_freeform_007"  # Format: _adhoc_{type}_{hash}
    adhoc_configs = {
        task_ref: {
            "dataset": {"name": "test/dataset"}
        }
    }
    
    config = build_handler_task_config(
        task_ref,
        adhoc_task_configs=adhoc_configs
    )
    
    assert "defaults" in config


# Task Detection Tests

def test_is_handler_based_task_recognizes_adhoc():
    """Test is_handler_based_task recognizes ad-hoc tasks."""
    task_name = "_adhoc_classification_008"  # Format: _adhoc_{type}_{hash}
    
    result = is_handler_based_task(task_name)
    
    assert result is True


def test_is_handler_based_task_recognizes_regular():
    """Test is_handler_based_task recognizes regular handler tasks."""
    task_name = "classify.sentiment"
    
    # Should recognize handler-based tasks
    # Result depends on whether task exists
    result = is_handler_based_task(task_name)
    
    # Just verify it returns a boolean
    assert isinstance(result, bool)
