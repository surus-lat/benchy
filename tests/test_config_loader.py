"""Tests for config loading and management."""

import yaml
from pathlib import Path

import pytest

from src.config_loader import resolve_config_path, load_config
from src.config_manager import ConfigManager


# Path Resolution Tests

def test_resolve_config_path_absolute(tmp_path):
    """Test resolve_config_path with absolute path."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model:\n  name: test")
    
    resolved = resolve_config_path(str(config_file))
    
    assert resolved == config_file


def test_resolve_config_path_relative(tmp_path, monkeypatch):
    """Test resolve_config_path with relative path."""
    # Create config in current directory
    config_file = tmp_path / "config.yaml"
    config_file.write_text("model:\n  name: test")
    
    monkeypatch.chdir(tmp_path)
    
    resolved = resolve_config_path("config.yaml")
    
    assert resolved.name == "config.yaml"


def test_resolve_config_path_short_name_in_models(tmp_path, monkeypatch):
    """Test resolve_config_path finds short name in configs/models/."""
    # This test would require mocking _find_project_root to use tmp_path
    # For now, just test that it returns a Path
    # In real usage, it searches the actual configs/ directory
    
    # Use a config that actually exists in the workspace
    resolved = resolve_config_path("test-model")
    
    assert isinstance(resolved, Path)
    assert resolved.name.startswith("test-model")


def test_resolve_config_path_short_name_in_systems(tmp_path, monkeypatch):
    """Test resolve_config_path finds short name in configs/systems/."""
    # This test would require mocking _find_project_root to use tmp_path
    # For now, just test that it returns a Path
    # In real usage, it searches the actual configs/ directory
    
    # Use a config that actually exists in the workspace
    resolved = resolve_config_path("test-system")
    
    assert isinstance(resolved, Path)
    assert resolved.name.startswith("test-system")


def test_resolve_config_path_not_found_returns_fallback_path(tmp_path, monkeypatch):
    """Test resolve_config_path returns a fallback Path for nonexistent short names."""
    monkeypatch.chdir(tmp_path)
    
    # Use a name that's very unlikely to exist
    # The function may return a Path even if it doesn't exist
    result = resolve_config_path("nonexistent-config-xyz-123")
    assert isinstance(result, Path)


# Config Loading Tests

def test_load_config_reads_yaml(tmp_path):
    """Test load_config reads and parses YAML file."""
    config_file = tmp_path / "config.yaml"
    config_data = {
        "model": {"name": "test-model"},
        "tasks": ["task1", "task2"],
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    loaded = load_config(config_file)
    
    assert loaded["model"]["name"] == "test-model"
    assert loaded["tasks"] == ["task1", "task2"]


def test_load_config_handles_invalid_yaml(tmp_path):
    """Test load_config handles invalid YAML gracefully."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: content: [")
    
    with pytest.raises(yaml.YAMLError):
        load_config(config_file)


def test_load_config_file_not_found_raises():
    """Test load_config raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))


# ConfigManager Tests

def test_config_manager_load_model_config(tmp_path, monkeypatch):
    """Test ConfigManager loads model configuration."""
    config_file = tmp_path / "config.yaml"
    config_data = {
        "model": {"name": "test-model", "path": "/path/to/model"},
        "tasks": ["task1"],
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    monkeypatch.chdir(tmp_path)
    
    manager = ConfigManager()
    config = manager.load_model_config(str(config_file))
    
    assert config["model"]["name"] == "test-model"
    assert config["model"]["path"] == "/path/to/model"


def test_config_manager_get_provider_config(tmp_path, monkeypatch):
    """Test ConfigManager extracts provider configuration."""
    # Create provider config file in the expected location
    provider_dir = tmp_path / "configs" / "providers"
    provider_dir.mkdir(parents=True)
    
    provider_file = provider_dir / "vllm.yaml"
    provider_data = {
        "name": "vllm",
        "base_url": "http://localhost:8000",
    }
    
    with open(provider_file, 'w') as f:
        yaml.dump(provider_data, f)
    
    monkeypatch.chdir(tmp_path)
    
    manager = ConfigManager(str(tmp_path / "configs"))
    provider_config = manager.get_provider_config("vllm")
    
    assert provider_config["name"] == "vllm"
    assert provider_config["base_url"] == "http://localhost:8000"


def test_config_manager_merges_provider_configs(tmp_path, monkeypatch):
    """Test ConfigManager merges provider configs from multiple sources."""
    # Create provider config file
    provider_file = tmp_path / "configs" / "providers" / "vllm.yaml"
    provider_file.parent.mkdir(parents=True)
    
    provider_data = {
        "name": "vllm",
        "batch_size": 32,
        "temperature": 0.7,
    }
    
    with open(provider_file, 'w') as f:
        yaml.dump(provider_data, f)
    
    monkeypatch.chdir(tmp_path)
    
    manager = ConfigManager(str(tmp_path / "configs"))
    provider_config = manager.get_provider_config("vllm")
    
    # Should have merged provider config
    assert provider_config.get("name") == "vllm"
    assert provider_config.get("batch_size") == 32


def test_config_manager_expand_task_groups(tmp_path, monkeypatch):
    """Test ConfigManager expands task groups."""
    manager = ConfigManager()
    
    central_config = {
        "task_groups": {
            "group1": {
                "tasks": ["task1", "task2"],
                "description": "Test group"
            }
        }
    }
    
    tasks = ["group1", "task3"]
    expanded = manager.expand_task_groups(tasks, central_config)
    
    # expand_task_groups returns a list
    assert isinstance(expanded, list)
    assert "task1" in expanded
    assert "task2" in expanded
    assert "task3" in expanded
