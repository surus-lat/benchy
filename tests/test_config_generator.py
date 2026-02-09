"""Tests for config generation from CLI arguments."""

import json
import yaml
from argparse import Namespace
from pathlib import Path

import pytest

from src.tasks.common.config_generator import (
    generate_config_from_cli,
    _build_dataset_config_from_args,
    _cleanup_empty_sections,
    validate_generated_config,
)


# Config Generation Tests

def test_generate_config_from_cli_with_adhoc_task(tmp_path, fake_cli_args):
    """Test generate_config_from_cli creates config for ad-hoc task."""
    output_file = tmp_path / "config.yaml"
    
    args = fake_cli_args(
        task_type="classification",
        dataset_name="test/dataset",
        dataset_input_field="text",
        dataset_output_field="label",
        dataset_labels='{"0": "No", "1": "Yes"}',
        system_prompt="Classify this",
        user_prompt_template="Text: {text}",
    )
    
    generate_config_from_cli(args, str(output_file))
    
    assert output_file.exists()
    
    with open(output_file) as f:
        config = yaml.safe_load(f)
    
    assert config["model"]["name"] == "test-model"
    assert "task_configs" in config
    assert len(config["task_configs"]) > 0


def test_generate_config_from_cli_with_dataset_override(tmp_path, fake_cli_args):
    """Test generate_config_from_cli creates config for dataset override."""
    output_file = tmp_path / "config.yaml"
    
    args = fake_cli_args(
        dataset_name="custom/dataset",
        dataset_source="huggingface",
        dataset_split="validation",
        dataset_input_field="question",
        dataset_output_field="answer",
    )
    
    generate_config_from_cli(args, str(output_file))
    
    assert output_file.exists()
    
    with open(output_file) as f:
        config = yaml.safe_load(f)
    
    # Dataset override goes into task_defaults
    assert "task_defaults" in config or "task_defaults_overrides" in config
    task_defaults_key = "task_defaults" if "task_defaults" in config else "task_defaults_overrides"
    assert config[task_defaults_key]["dataset"]["name"] == "custom/dataset"


def test_generate_config_from_cli_includes_model_info(tmp_path, fake_cli_args):
    """Test generate_config_from_cli includes model information."""
    output_file = tmp_path / "config.yaml"
    
    args = fake_cli_args(
        model_name="gpt-4",
        model_path="/path/to/model",
        provider="openai",
        base_url="https://api.openai.com",
        batch_size=10,
    )
    
    generate_config_from_cli(args, str(output_file))
    
    with open(output_file) as f:
        config = yaml.safe_load(f)
    
    assert config["model"]["name"] == "gpt-4"
    assert config["model"]["path"] == "/path/to/model"


def test_generate_config_from_cli_includes_provider(tmp_path, fake_cli_args):
    """Test generate_config_from_cli includes provider configuration."""
    output_file = tmp_path / "config.yaml"
    
    args = fake_cli_args(
        provider="vllm",
        base_url="http://localhost:8000",
        batch_size=32,
    )
    
    generate_config_from_cli(args, str(output_file))
    
    with open(output_file) as f:
        config = yaml.safe_load(f)
    
    # Provider type is stored separately
    assert config["provider_type"] == "vllm"
    # batch_size should be in task_defaults
    assert "task_defaults" in config


def test_generate_config_from_cli_cleans_empty_sections(tmp_path, fake_cli_args):
    """Test generate_config_from_cli removes empty sections."""
    output_file = tmp_path / "config.yaml"
    
    args = fake_cli_args(
        # All defaults, no specific overrides
    )
    
    generate_config_from_cli(args, str(output_file))
    
    with open(output_file) as f:
        config = yaml.safe_load(f)
    
    # Empty sections should be removed
    assert "provider" not in config or config["provider"]
    assert "path" not in config.get("model", {})


# Config Reuse Tests

def test_generated_config_is_valid_yaml(tmp_path, fake_cli_args):
    """Test generated config is valid YAML."""
    output_file = tmp_path / "config.yaml"
    
    args = fake_cli_args(
        task_type="freeform",
        dataset_name="test/dataset",
    )
    
    generate_config_from_cli(args, str(output_file))
    
    # Should not raise
    with open(output_file) as f:
        config = yaml.safe_load(f)
    
    assert isinstance(config, dict)


def test_generated_config_preserves_dataset_config(tmp_path, fake_cli_args):
    """Test generated config preserves dataset configuration."""
    output_file = tmp_path / "config.yaml"
    
    args = fake_cli_args(
        task_type="classification",
        dataset_name="test/dataset",
        dataset_source="huggingface",
        dataset_split="validation",
        dataset_input_field="question",
        dataset_output_field="answer",
        dataset_id_field="id",
        dataset_label_field="label",
        dataset_labels='{"0": "No", "1": "Yes"}',
        multimodal_input=True,
        multimodal_image_field="image",
    )
    
    generate_config_from_cli(args, str(output_file))
    
    with open(output_file) as f:
        config = yaml.safe_load(f)
    
    # Find the ad-hoc task config
    task_configs = config.get("task_configs", {})
    adhoc_task = next(iter(task_configs.values()))
    
    assert adhoc_task["dataset"]["name"] == "test/dataset"
    assert adhoc_task["dataset"]["input_field"] == "question"
    assert adhoc_task["dataset"]["multimodal_input"] is True


def test_generated_config_preserves_prompts(tmp_path, fake_cli_args):
    """Test generated config preserves custom prompts."""
    output_file = tmp_path / "config.yaml"
    
    args = fake_cli_args(
        task_type="freeform",
        dataset_name="test/dataset",
        system_prompt="You are a helpful assistant",
        user_prompt_template="Question: {text}\nAnswer:",
    )
    
    generate_config_from_cli(args, str(output_file))
    
    with open(output_file) as f:
        config = yaml.safe_load(f)
    
    task_configs = config.get("task_configs", {})
    adhoc_task = next(iter(task_configs.values()))
    
    assert adhoc_task["system_prompt"] == "You are a helpful assistant"
    assert adhoc_task["user_prompt_template"] == "Question: {text}\nAnswer:"


# Validation Tests

def test_validate_generated_config_requires_model_name():
    """Test validate_generated_config requires model.name."""
    config = {
        "model": {},
        "tasks": ["task1"],
    }
    
    errors = validate_generated_config(config)
    
    assert len(errors) > 0
    assert any("model.name" in err for err in errors)


def test_validate_generated_config_requires_tasks():
    """Test validate_generated_config requires tasks or task_configs."""
    config = {
        "model": {"name": "test-model"},
    }
    
    errors = validate_generated_config(config)
    
    assert len(errors) > 0
    assert any("tasks" in err.lower() for err in errors)


def test_validate_generated_config_valid():
    """Test validate_generated_config returns empty for valid config."""
    config = {
        "model": {"name": "test-model"},
        "tasks": ["task1"],
    }
    
    errors = validate_generated_config(config)
    
    assert len(errors) == 0


# Cleanup Tests

def test_cleanup_empty_sections_removes_empty_dicts():
    """Test _cleanup_empty_sections removes empty dictionaries."""
    config = {
        "model": {"name": "test"},
        "empty_section": {},
        "nested": {
            "has_value": "value",
            "empty_nested": {},
        }
    }
    
    cleaned = _cleanup_empty_sections(config)
    
    assert "empty_section" not in cleaned
    assert "empty_nested" not in cleaned["nested"]
    assert cleaned["nested"]["has_value"] == "value"


def test_cleanup_empty_sections_preserves_non_empty():
    """Test _cleanup_empty_sections preserves non-empty sections."""
    config = {
        "model": {"name": "test", "path": "/path"},
        "provider": {"name": "vllm"},
        "tasks": ["task1", "task2"],
    }
    
    cleaned = _cleanup_empty_sections(config)
    
    assert cleaned == config  # Nothing should be removed
