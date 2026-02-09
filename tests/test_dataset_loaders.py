"""Tests for dataset mixin classes."""

import json
from pathlib import Path

import pytest

from src.tasks.common.dataset_loaders import (
    CachedDatasetMixin,
    CachedTSVMixin,
    CachedCSVMixin,
)
from src.tasks.common.utils.dataset_utils import save_to_jsonl


# Test implementations of mixins

class SimpleCachedDatasetHandler(CachedDatasetMixin):
    """Test handler using CachedDatasetMixin."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.dataset_file = "test.jsonl"
        self.dataset_name = "test/dataset"
        self._download_called = False
    
    def _download_and_cache(self, output_path: Path):
        """Fake download implementation."""
        self._download_called = True
        samples = [
            {"id": "1", "text": "Sample 1", "expected": "Output 1"},
            {"id": "2", "text": "Sample 2", "expected": "Output 2"},
        ]
        save_to_jsonl(samples, output_path)


class SimpleCachedTSVHandler(CachedTSVMixin):
    """Test handler using CachedTSVMixin."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.dataset_file = "test.jsonl"
        self.tsv_filename = "test.tsv"
        self._download_called = False
    
    def _download_tsv_and_cache(self, output_path: Path):
        """Fake TSV download implementation."""
        self._download_called = True
        samples = [
            {"id": "1", "question": "Q1", "answer": "A1"},
            {"id": "2", "question": "Q2", "answer": "A2"},
        ]
        save_to_jsonl(samples, output_path)


class SimpleCachedCSVHandler(CachedCSVMixin):
    """Test handler using CachedCSVMixin."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.dataset_file = "test.jsonl"
        self.csv_filename = "test.csv"
        self._download_called = False
    
    def _download_csv_and_cache(self, output_path: Path):
        """Fake CSV download implementation."""
        self._download_called = True
        samples = [
            {"id": "1", "text": "Text 1", "label": 0},
            {"id": "2", "text": "Text 2", "label": 1},
        ]
        save_to_jsonl(samples, output_path)


# CachedDatasetMixin Tests

def test_cached_dataset_mixin_downloads_and_caches(tmp_path):
    """Test CachedDatasetMixin downloads when cache doesn't exist."""
    handler = SimpleCachedDatasetHandler(tmp_path)
    
    loaded = handler.load_dataset()
    
    assert handler._download_called is True
    assert len(loaded) == 2
    assert loaded[0]["id"] == "1"
    assert (tmp_path / "test.jsonl").exists()


def test_cached_dataset_mixin_uses_cache_when_exists(tmp_path):
    """Test CachedDatasetMixin uses cache when it exists."""
    # Pre-create cached file
    cache_file = tmp_path / "test.jsonl"
    cached_samples = [
        {"id": "cached_1", "text": "Cached sample"},
    ]
    save_to_jsonl(cached_samples, cache_file)
    
    handler = SimpleCachedDatasetHandler(tmp_path)
    
    loaded = handler.load_dataset()
    
    assert handler._download_called is False  # Should not download
    assert len(loaded) == 1
    assert loaded[0]["id"] == "cached_1"


def test_cached_dataset_mixin_calls_download_and_cache(tmp_path):
    """Test CachedDatasetMixin calls _download_and_cache with correct path."""
    handler = SimpleCachedDatasetHandler(tmp_path)
    
    handler.load_dataset()
    
    assert handler._download_called is True
    assert (tmp_path / "test.jsonl").exists()


# CachedTSVMixin Tests

def test_cached_tsv_mixin_loads_tsv(tmp_path):
    """Test CachedTSVMixin downloads and converts TSV."""
    handler = SimpleCachedTSVHandler(tmp_path)
    
    loaded = handler.load_dataset()
    
    assert handler._download_called is True
    assert len(loaded) == 2
    assert loaded[0]["question"] == "Q1"


def test_cached_tsv_mixin_uses_cache(tmp_path):
    """Test CachedTSVMixin uses cache when it exists."""
    # Pre-create cached file
    cache_file = tmp_path / "test.jsonl"
    cached_samples = [
        {"id": "cached_1", "question": "Cached Q", "answer": "Cached A"},
    ]
    save_to_jsonl(cached_samples, cache_file)
    
    handler = SimpleCachedTSVHandler(tmp_path)
    
    loaded = handler.load_dataset()
    
    assert handler._download_called is False
    assert len(loaded) == 1
    assert loaded[0]["question"] == "Cached Q"


# CachedCSVMixin Tests

def test_cached_csv_mixin_loads_csv(tmp_path):
    """Test CachedCSVMixin downloads and converts CSV."""
    handler = SimpleCachedCSVHandler(tmp_path)
    
    loaded = handler.load_dataset()
    
    assert handler._download_called is True
    assert len(loaded) == 2
    assert loaded[0]["text"] == "Text 1"


def test_cached_csv_mixin_uses_cache(tmp_path):
    """Test CachedCSVMixin uses cache when it exists."""
    # Pre-create cached file
    cache_file = tmp_path / "test.jsonl"
    cached_samples = [
        {"id": "cached_1", "text": "Cached text", "label": 2},
    ]
    save_to_jsonl(cached_samples, cache_file)
    
    handler = SimpleCachedCSVHandler(tmp_path)
    
    loaded = handler.load_dataset()
    
    assert handler._download_called is False
    assert len(loaded) == 1
    assert loaded[0]["label"] == 2
