"""Tests for CLI argument parsing and processing."""

from argparse import Namespace

import pytest

from src.benchy_cli_eval import (
    _build_dataset_config_from_args,
    _build_adhoc_task_config,
)


# Dataset Config Building Tests

def test_build_dataset_config_from_args_basic():
    """Test _build_dataset_config_from_args with basic dataset name."""
    args = Namespace(
        dataset_name="test/dataset",
        dataset_source="auto",
        dataset_split="test",
        dataset_input_field=None,
        dataset_output_field=None,
        dataset_id_field=None,
        dataset_label_field=None,
        dataset_labels=None,
        dataset_choices_field=None,
        dataset_schema_field=None,
        dataset_schema_path=None,
        dataset_schema_json=None,
        multimodal_input=False,
        multimodal_image_field="image_path",
    )
    
    config = _build_dataset_config_from_args(args)
    
    assert config["name"] == "test/dataset"
    assert config["source"] == "auto"
    assert config["split"] == "test"


def test_build_dataset_config_from_args_with_field_mappings():
    """Test _build_dataset_config_from_args with custom field mappings."""
    args = Namespace(
        dataset_name="test/dataset",
        dataset_source="auto",
        dataset_split="test",
        dataset_input_field="question",
        dataset_output_field="answer",
        dataset_id_field="sample_id",
        dataset_label_field="label_value",
        dataset_labels=None,
        dataset_choices_field=None,
        dataset_schema_field=None,
        dataset_schema_path=None,
        dataset_schema_json=None,
        multimodal_input=False,
        multimodal_image_field="image_path",
    )
    
    config = _build_dataset_config_from_args(args)
    
    assert config["input_field"] == "question"
    assert config["output_field"] == "answer"
    assert config["id_field"] == "sample_id"
    assert config["label_field"] == "label_value"


def test_build_dataset_config_from_args_with_labels():
    """Test _build_dataset_config_from_args with classification labels."""
    args = Namespace(
        dataset_name="test/dataset",
        dataset_source="auto",
        dataset_split="test",
        dataset_input_field=None,
        dataset_output_field=None,
        dataset_id_field=None,
        dataset_label_field=None,
        dataset_labels='{"0": "Negative", "1": "Positive"}',
        dataset_choices_field="choices",
        dataset_schema_field=None,
        dataset_schema_path=None,
        dataset_schema_json=None,
        multimodal_input=False,
        multimodal_image_field="image_path",
    )
    
    config = _build_dataset_config_from_args(args)
    
    assert config["labels"] == '{"0": "Negative", "1": "Positive"}'
    assert config["choices_field"] == "choices"


def test_build_dataset_config_from_args_with_schema():
    """Test _build_dataset_config_from_args with schema configuration."""
    args = Namespace(
        dataset_name="test/dataset",
        dataset_source="auto",
        dataset_split="test",
        dataset_input_field=None,
        dataset_output_field=None,
        dataset_id_field=None,
        dataset_label_field=None,
        dataset_labels=None,
        dataset_choices_field=None,
        dataset_schema_field="json_schema",
        dataset_schema_path="/path/to/schema.json",
        dataset_schema_json='{"type": "object"}',
        multimodal_input=False,
        multimodal_image_field="image_path",
    )
    
    config = _build_dataset_config_from_args(args)
    
    assert config["schema_field"] == "json_schema"
    assert config["schema_path"] == "/path/to/schema.json"
    assert config["schema_json"] == '{"type": "object"}'


def test_build_dataset_config_from_args_with_multimodal():
    """Test _build_dataset_config_from_args with multimodal settings."""
    args = Namespace(
        dataset_name="test/dataset",
        dataset_source="auto",
        dataset_split="test",
        dataset_input_field=None,
        dataset_output_field=None,
        dataset_id_field=None,
        dataset_label_field=None,
        dataset_labels=None,
        dataset_choices_field=None,
        dataset_schema_field=None,
        dataset_schema_path=None,
        dataset_schema_json=None,
        multimodal_input=True,
        multimodal_image_field="image",
    )
    
    config = _build_dataset_config_from_args(args)
    
    assert config["multimodal_input"] is True
    assert config["multimodal_image_field"] == "image"


def test_build_dataset_config_from_args_empty_when_no_dataset(fake_cli_args):
    """Test _build_dataset_config_from_args returns minimal dict when no dataset."""
    args = fake_cli_args(dataset_name=None)
    
    config = _build_dataset_config_from_args(args)
    
    # Returns defaults even without dataset_name
    assert isinstance(config, dict)


# Ad-hoc Task Config Building Tests

def test_build_adhoc_task_config_classification():
    """Test _build_adhoc_task_config for classification task."""
    dataset_config = {
        "name": "test/dataset",
        "labels": '{"0": "No", "1": "Yes"}',
    }
    
    config = _build_adhoc_task_config(
        "classification",
        dataset_config,
        system_prompt="Classify this",
        user_prompt_template="Text: {text}"
    )
    
    # Config doesn't include task_type, that's passed separately
    assert "dataset" in config
    assert config["system_prompt"] == "Classify this"
    assert config["user_prompt_template"] == "Text: {text}"


def test_build_adhoc_task_config_structured():
    """Test _build_adhoc_task_config for structured extraction task."""
    dataset_config = {
        "name": "test/dataset",
        "schema_path": "/path/to/schema.json",
    }
    
    config = _build_adhoc_task_config(
        "structured",
        dataset_config,
    )
    
    # Config doesn't include task_type, that's passed separately
    assert "dataset" in config
    assert config["dataset"]["schema_path"] == "/path/to/schema.json"


def test_build_adhoc_task_config_freeform():
    """Test _build_adhoc_task_config for freeform generation task."""
    dataset_config = {
        "name": "test/dataset",
    }
    
    config = _build_adhoc_task_config(
        "freeform",
        dataset_config,
    )
    
    # Config doesn't include task_type, that's passed separately
    assert "dataset" in config


def test_build_adhoc_task_config_validates_config():
    """Test _build_adhoc_task_config validates the generated config."""
    dataset_config = {
        "name": "test/dataset",
        "labels": '{"0": "No", "1": "Yes"}',  # Required for classification
    }
    
    # Should not raise for valid config
    config = _build_adhoc_task_config("classification", dataset_config)
    
    assert "dataset" in config


def test_build_adhoc_task_config_applies_defaults():
    """Test _build_adhoc_task_config applies task type defaults."""
    dataset_config = {
        "name": "test/dataset",
        "labels": '{"0": "No", "1": "Yes"}',
    }
    
    config = _build_adhoc_task_config("classification", dataset_config)
    
    # Should have defaults applied
    assert "input_field" in config["dataset"]
    assert "output_field" in config["dataset"]


def test_build_adhoc_task_config_raises_on_invalid():
    """Test _build_adhoc_task_config raises on invalid task type."""
    dataset_config = {"name": "test/dataset"}
    
    with pytest.raises(ValueError, match="Invalid task"):
        _build_adhoc_task_config("invalid_type", dataset_config)
