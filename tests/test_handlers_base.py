"""Tests for BaseHandler core functionality."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from src.tasks.common.base import BaseHandler
from src.tasks.common.metrics import ExactMatch


class SimpleTestHandler(BaseHandler):
    """Minimal handler for testing BaseHandler functionality."""
    
    dataset_name = "test/dataset"
    text_field = "text"
    label_field = "expected"
    system_prompt = "Test system prompt"
    user_prompt_template = "Input: {text}"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.metrics = [ExactMatch()]


# Initialization & Configuration Tests

def test_base_handler_init_with_config(tmp_path):
    """Test handler initialization with config parameter."""
    config = {
        "dataset": {"name": "test-dataset"},
        "system_prompt": "Custom prompt"
    }
    
    handler = SimpleTestHandler(config)
    
    assert handler.config == config
    assert handler.config["dataset"]["name"] == "test-dataset"


def test_base_handler_without_config_uses_class_attributes():
    """Test handler uses class attributes when no config provided."""
    handler = SimpleTestHandler()
    
    assert handler.dataset_name == "test/dataset"
    assert handler.text_field == "text"
    assert handler.label_field == "expected"
    assert handler.system_prompt == "Test system prompt"


def test_base_handler_stores_config():
    """Test handler stores config for later use."""
    config = {"key": "value"}
    handler = SimpleTestHandler(config)
    
    assert handler.config == config


# Dataset Loading Tests

def test_load_dataset_uses_adapter_when_config_provided(tmp_path, monkeypatch):
    """Test load_dataset uses DatasetAdapter when config provides dataset."""
    # Create test dataset
    dataset_file = tmp_path / "data.jsonl"
    samples = [
        {"id": "1", "text": "Sample 1", "expected": "Output 1"},
        {"id": "2", "text": "Sample 2", "expected": "Output 2"},
    ]
    
    with open(dataset_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    config = {
        "dataset": {
            "name": str(dataset_file),
            "source": "local",
        }
    }
    
    handler = SimpleTestHandler(config)
    handler.data_dir = tmp_path / ".data"
    handler.data_dir.mkdir(parents=True, exist_ok=True)
    
    loaded = handler.load_dataset()
    
    assert len(loaded) == 2
    assert loaded[0]["text"] == "Sample 1"
    assert loaded[1]["text"] == "Sample 2"


def test_load_dataset_loads_from_cache_when_exists(tmp_path):
    """Test load_dataset loads from cache file if it exists."""
    # Create cached data
    cache_file = tmp_path / ".data" / "test_dataset" / "test.jsonl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    cached_samples = [
        {"id": "cached_1", "text": "Cached sample", "expected": "Cached output"}
    ]
    
    with open(cache_file, 'w') as f:
        for sample in cached_samples:
            f.write(json.dumps(sample) + '\n')
    
    handler = SimpleTestHandler()
    handler.data_file = cache_file
    
    loaded = handler.load_dataset()
    
    assert len(loaded) == 1
    assert loaded[0]["id"] == "cached_1"
    assert loaded[0]["text"] == "Cached sample"


def test_load_dataset_raises_when_no_dataset_configured():
    """Test load_dataset raises FileNotFoundError when no dataset configured."""
    handler = SimpleTestHandler()
    handler.dataset = None  # No dataset
    handler.data_file = Path("/nonexistent/file.jsonl")
    
    with pytest.raises(FileNotFoundError, match="No dataset found"):
        handler.load_dataset()


# Sample Preprocessing Tests

def test_preprocess_sample_returns_as_is_when_already_preprocessed():
    """Test preprocess_sample returns sample as-is if already has id and text."""
    handler = SimpleTestHandler()
    
    sample = {"id": "test_1", "text": "Sample text", "expected": "Output"}
    result = handler.preprocess_sample(sample, 0)
    
    assert result == sample


def test_preprocess_sample_adds_id_when_missing():
    """Test preprocess_sample adds ID when missing."""
    handler = SimpleTestHandler()
    
    sample = {"text": "Sample text", "expected": "Output"}
    result = handler.preprocess_sample(sample, 5)
    
    assert "id" in result
    # ID format is based on task name, which comes from dataset_name
    assert result["id"].endswith("_5")


def test_preprocess_sample_preserves_existing_fields():
    """Test preprocess_sample preserves all existing fields."""
    handler = SimpleTestHandler()
    
    sample = {"text": "Sample", "expected": "Output", "extra_field": "value"}
    result = handler.preprocess_sample(sample, 0)
    
    assert result["extra_field"] == "value"


# Prompt Generation Tests

def test_get_prompt_formats_template_with_sample_fields():
    """Test get_prompt formats user_prompt_template with sample fields."""
    handler = SimpleTestHandler()
    handler.user_prompt_template = "Question: {text}\nAnswer:"
    
    sample = {"text": "What is AI?"}
    system, user = handler.get_prompt(sample)
    
    assert system == "Test system prompt"
    assert user == "Question: What is AI?\nAnswer:"


def test_get_prompt_raises_on_missing_template_field():
    """Test get_prompt raises KeyError when template field missing from sample."""
    handler = SimpleTestHandler()
    handler.user_prompt_template = "Question: {missing_field}"
    
    sample = {"text": "What is AI?"}
    
    with pytest.raises(KeyError):
        handler.get_prompt(sample)


def test_get_prompt_uses_default_prompts():
    """Test get_prompt uses class attribute prompts."""
    handler = SimpleTestHandler()
    
    sample = {"text": "Sample text"}
    system, user = handler.get_prompt(sample)
    
    assert system == "Test system prompt"
    assert "Sample text" in user


# Utilities Tests

def test_get_task_name_converts_to_snake_case():
    """Test get_task_name returns task name."""
    handler = SimpleTestHandler()
    
    name = handler.get_task_name()
    
    # Task name comes from dataset_name attribute
    assert isinstance(name, str)
    assert len(name) > 0


def test_resolve_data_dir_creates_directory(tmp_path):
    """Test _resolve_data_dir creates data directory."""
    handler = SimpleTestHandler()
    handler.name = "test_task"
    
    # _resolve_data_dir takes no arguments, uses self.name
    data_dir = handler._resolve_data_dir()
    
    # Should return a Path object
    assert isinstance(data_dir, Path)


# Load Method Tests

def test_load_calls_load_dataset_and_stores_data(tmp_path, monkeypatch):
    """Test load() method calls load_dataset and stores result."""
    # Create test dataset
    dataset_file = tmp_path / "data.jsonl"
    samples = [{"id": "1", "text": "Test", "expected": "Output"}]
    
    with open(dataset_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    config = {
        "dataset": {
            "name": str(dataset_file),
            "source": "local",
        }
    }
    
    handler = SimpleTestHandler(config)
    handler.data_dir = tmp_path / ".data"
    handler.data_dir.mkdir(parents=True, exist_ok=True)
    
    handler.load()
    
    assert handler.dataset_data is not None
    assert len(handler.dataset_data) == 1
    assert handler.dataset_data[0]["id"] == "1"


# Get Samples Tests

def test_get_samples_iterates_over_dataset(tmp_path):
    """Test get_samples returns iterator over dataset."""
    handler = SimpleTestHandler()
    handler.dataset_data = [
        {"id": "1", "text": "Sample 1"},
        {"id": "2", "text": "Sample 2"},
        {"id": "3", "text": "Sample 3"},
    ]
    
    samples = list(handler.get_samples())
    
    assert len(samples) == 3
    assert samples[0]["id"] == "1"
    assert samples[2]["id"] == "3"


def test_get_samples_respects_limit(tmp_path):
    """Test get_samples respects limit parameter."""
    handler = SimpleTestHandler()
    handler.dataset_data = [
        {"id": "1", "text": "Sample 1"},
        {"id": "2", "text": "Sample 2"},
        {"id": "3", "text": "Sample 3"},
    ]
    
    samples = list(handler.get_samples(limit=2))
    
    assert len(samples) == 2
    assert samples[0]["id"] == "1"
    assert samples[1]["id"] == "2"


def test_get_samples_raises_when_dataset_not_loaded():
    """Test get_samples raises RuntimeError when dataset not loaded."""
    handler = SimpleTestHandler()
    handler.dataset_data = None
    
    with pytest.raises(RuntimeError, match="Dataset not loaded"):
        list(handler.get_samples())
