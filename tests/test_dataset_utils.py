"""Tests for dataset utility functions."""

import json

import pytest

from src.tasks.common.utils.dataset_utils import (
    load_jsonl_dataset,
    save_to_jsonl,
    download_huggingface_dataset,
    iterate_samples,
)


# JSONL Operations Tests

def test_load_jsonl_dataset_reads_file(tmp_path):
    """Test load_jsonl_dataset reads JSONL file."""
    dataset_file = tmp_path / "data.jsonl"
    samples = [
        {"id": "1", "text": "Sample 1"},
        {"id": "2", "text": "Sample 2"},
        {"id": "3", "text": "Sample 3"},
    ]
    
    with open(dataset_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    loaded = load_jsonl_dataset(dataset_file)
    
    assert len(loaded) == 3
    assert loaded[0]["id"] == "1"
    assert loaded[2]["text"] == "Sample 3"


def test_load_jsonl_dataset_empty_file(tmp_path):
    """Test load_jsonl_dataset handles empty file."""
    dataset_file = tmp_path / "empty.jsonl"
    dataset_file.touch()
    
    loaded = load_jsonl_dataset(dataset_file)
    
    assert len(loaded) == 0


def test_load_jsonl_dataset_malformed_json(tmp_path):
    """Test load_jsonl_dataset handles malformed JSON lines."""
    dataset_file = tmp_path / "malformed.jsonl"
    
    with open(dataset_file, 'w') as f:
        f.write('{"id": "1", "text": "Valid"}\n')
        f.write('invalid json line\n')
        f.write('{"id": "2", "text": "Also valid"}\n')
    
    with pytest.raises(json.JSONDecodeError):
        load_jsonl_dataset(dataset_file)


def test_save_to_jsonl_writes_file(tmp_path):
    """Test save_to_jsonl writes samples to JSONL file."""
    output_file = tmp_path / "output.jsonl"
    samples = [
        {"id": "1", "text": "Sample 1"},
        {"id": "2", "text": "Sample 2"},
    ]
    
    save_to_jsonl(samples, output_file)
    
    assert output_file.exists()
    
    # Verify contents
    loaded = load_jsonl_dataset(output_file)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "1"


def test_save_to_jsonl_creates_directory(tmp_path):
    """Test save_to_jsonl creates parent directories."""
    output_file = tmp_path / "subdir" / "nested" / "output.jsonl"
    samples = [{"id": "1", "text": "Sample"}]
    
    save_to_jsonl(samples, output_file)
    
    assert output_file.exists()
    assert output_file.parent.exists()


# HuggingFace Download Tests

def test_download_huggingface_dataset_success(monkeypatch):
    """Test download_huggingface_dataset downloads and converts dataset."""
    fake_dataset = [
        {"id": "1", "text": "Sample 1"},
        {"id": "2", "text": "Sample 2"},
    ]
    
    def fake_load_dataset(name, split=None, **kwargs):
        class FakeDataset:
            def __iter__(self):
                return iter(fake_dataset)
        return FakeDataset()
    
    monkeypatch.setattr(
        "src.tasks.common.utils.dataset_utils.load_dataset",
        fake_load_dataset
    )
    
    downloaded = download_huggingface_dataset("test/dataset", split="test")
    
    assert len(downloaded) == 2
    assert downloaded[0]["id"] == "1"


def test_download_huggingface_dataset_with_split(monkeypatch):
    """Test download_huggingface_dataset uses specified split."""
    split_used = None
    
    def fake_load_dataset(name, split=None, **kwargs):
        nonlocal split_used
        split_used = split
        class FakeDataset:
            def __iter__(self):
                return iter([{"id": "1"}])
        return FakeDataset()
    
    monkeypatch.setattr(
        "src.tasks.common.utils.dataset_utils.load_dataset",
        fake_load_dataset
    )
    
    download_huggingface_dataset("test/dataset", split="validation")
    
    assert split_used == "validation"


def test_download_huggingface_dataset_fallback_to_train(monkeypatch):
    """Test download_huggingface_dataset falls back to train split."""
    attempts = []
    
    def fake_load_dataset(name, split=None, **kwargs):
        attempts.append(split)
        if split == "test":
            raise ValueError("Split not found")
        class FakeDataset:
            def __iter__(self):
                return iter([{"id": "1"}])
        return FakeDataset()
    
    monkeypatch.setattr(
        "src.tasks.common.utils.dataset_utils.load_dataset",
        fake_load_dataset
    )
    
    try:
        download_huggingface_dataset("test/dataset", split="test")
    except Exception:
        pass

    assert attempts == ["test", "train"]


def test_download_huggingface_dataset_failure_raises(monkeypatch):
    """Test download_huggingface_dataset raises on failure."""
    def fake_load_dataset(name, split=None, **kwargs):
        raise Exception("Download failed")
    
    monkeypatch.setattr(
        "src.tasks.common.utils.dataset_utils.load_dataset",
        fake_load_dataset
    )
    
    with pytest.raises(Exception):
        download_huggingface_dataset("test/dataset")


# Sample Iteration Tests

def test_iterate_samples_no_limit():
    """Test iterate_samples yields all samples without limit."""
    samples = [
        {"id": "1"},
        {"id": "2"},
        {"id": "3"},
    ]
    
    result = list(iterate_samples(samples))
    
    assert len(result) == 3


def test_iterate_samples_with_limit():
    """Test iterate_samples respects limit parameter."""
    samples = [
        {"id": "1"},
        {"id": "2"},
        {"id": "3"},
        {"id": "4"},
        {"id": "5"},
    ]
    
    result = list(iterate_samples(samples, limit=3))
    
    assert len(result) == 3
    assert result[0]["id"] == "1"
    assert result[2]["id"] == "3"


def test_iterate_samples_empty_list():
    """Test iterate_samples handles empty list."""
    samples = []
    
    result = list(iterate_samples(samples))
    
    assert len(result) == 0


def test_iterate_samples_limit_larger_than_dataset():
    """Test iterate_samples handles limit larger than dataset."""
    samples = [
        {"id": "1"},
        {"id": "2"},
    ]
    
    result = list(iterate_samples(samples, limit=10))
    
    assert len(result) == 2  # Only 2 samples available
