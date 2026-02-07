"""Mixins for tasks that cache datasets to local files.

These mixins provide automatic download-and-cache behavior using existing
dataset_utils functions. Subclasses only implement the unique transformation
logic in their _download_and_cache() method.

This eliminates ~40-60 lines of boilerplate per task.
"""

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from .utils.dataset_utils import load_jsonl_dataset

logger = logging.getLogger(__name__)


class CachedDatasetMixin:
    """Mixin for tasks that cache datasets to local JSONL files.
    
    Provides automatic download-and-cache behavior. Subclasses only implement
    the unique transformation logic in _download_and_cache().
    
    Required attributes (set in subclass):
        dataset_file: str - Name of cached file (e.g., "test.jsonl")
        data_dir: Path - Directory for cached data
    
    Example:
        class MyTask(CachedDatasetMixin, MultipleChoiceHandler):
            dataset_file = "test.jsonl"
            
            def _download_and_cache(self, output_path: Path):
                raw = download_huggingface_dataset(...)
                processed = [transform(sample) for sample in raw]
                save_to_jsonl(processed, output_path)
    """
    
    # These must be set in subclass
    dataset_file: str
    data_dir: Path
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from cached file, auto-downloading if needed.
        
        Uses the existing load_jsonl_dataset utility and triggers
        _download_and_cache if the file doesn't exist.
        """
        file_path = self.data_dir / self.dataset_file
        
        if not file_path.exists():
            dataset_name = getattr(self, 'dataset_name', 'dataset')
            logger.info(f"Cached file not found, downloading {dataset_name}")
            self._download_and_cache(file_path)
        
        # Use existing utility!
        return load_jsonl_dataset(file_path)
    
    @abstractmethod
    def _download_and_cache(self, output_path: Path):
        """Download and transform dataset, saving to output_path.
        
        This is the ONLY method subclasses need to implement.
        Should use save_to_jsonl() from dataset_utils to save.
        
        Args:
            output_path: Path where processed JSONL should be saved
        
        Example:
            def _download_and_cache(self, output_path: Path):
                raw = download_huggingface_dataset(self.dataset_name, self.split)
                processed = [self._transform(s) for s in raw]
                save_to_jsonl(processed, output_path)
        """
        pass


class CachedTSVMixin:
    """Mixin for tasks that cache TSV datasets (like MGSM).
    
    TSV files are downloaded and converted to JSONL for consistent handling.
    
    Required attributes (set in subclass):
        dataset_file: str - Name of cached JSONL file (e.g., "test.jsonl")
        data_dir: Path - Directory for cached data
        tsv_filename: str - Name of TSV file to download (e.g., "mgsm_es.tsv")
    
    Example:
        class MgsmTask(CachedTSVMixin, FreeformHandler):
            dataset_file = "test.jsonl"
            tsv_filename = "mgsm_es.tsv"
            
            def _download_tsv_and_cache(self, output_path: Path):
                tsv_path = hf_hub_download(self.dataset_name, self.tsv_filename)
                processed = []
                with open(tsv_path) as f:
                    for line in f:
                        question, answer = line.strip().split('\\t')
                        processed.append({"text": question, "expected": answer})
                save_to_jsonl(processed, output_path)
    """
    
    dataset_file: str
    data_dir: Path
    tsv_filename: str
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load from cached JSONL (TSV already converted)."""
        file_path = self.data_dir / self.dataset_file
        
        if not file_path.exists():
            logger.info(f"Cached file not found, downloading TSV: {self.tsv_filename}")
            self._download_tsv_and_cache(file_path)
        
        return load_jsonl_dataset(file_path)
    
    @abstractmethod
    def _download_tsv_and_cache(self, output_path: Path):
        """Download TSV, transform, and save as JSONL.
        
        Args:
            output_path: Path where processed JSONL should be saved
        
        Example:
            def _download_tsv_and_cache(self, output_path: Path):
                from huggingface_hub import hf_hub_download
                
                tsv_path = hf_hub_download(
                    repo_id=self.dataset_name,
                    filename=self.tsv_filename,
                    cache_dir=str(self.data_dir / "cache"),
                )
                
                processed = []
                with open(tsv_path, "r") as f:
                    for line in f:
                        parts = line.strip().split('\\t')
                        processed.append(self._transform(parts))
                
                save_to_jsonl(processed, output_path)
        """
        pass


class CachedCSVMixin:
    """Mixin for tasks that cache CSV datasets (like WNLI-es).
    
    CSV files are downloaded and converted to JSONL for consistent handling.
    
    Required attributes (set in subclass):
        dataset_file: str - Name of cached JSONL file (e.g., "validation.jsonl")
        data_dir: Path - Directory for cached data
        csv_filename: str - Name of CSV file to download (e.g., "wnli-dev-es.csv")
    
    Example:
        class WnliTask(CachedCSVMixin, MultipleChoiceHandler):
            dataset_file = "validation.jsonl"
            csv_filename = "wnli-dev-es.csv"
            
            def _download_csv_and_cache(self, output_path: Path):
                csv_path = hf_hub_download(self.dataset_name, self.csv_filename)
                processed = []
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        processed.append(self._transform(row))
                save_to_jsonl(processed, output_path)
    """
    
    dataset_file: str
    data_dir: Path
    csv_filename: str
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load from cached JSONL (CSV already converted)."""
        file_path = self.data_dir / self.dataset_file
        
        if not file_path.exists():
            logger.info(f"Cached file not found, downloading CSV: {self.csv_filename}")
            self._download_csv_and_cache(file_path)
        
        return load_jsonl_dataset(file_path)
    
    @abstractmethod
    def _download_csv_and_cache(self, output_path: Path):
        """Download CSV, transform, and save as JSONL.
        
        Args:
            output_path: Path where processed JSONL should be saved
        
        Example:
            def _download_csv_and_cache(self, output_path: Path):
                import csv
                from huggingface_hub import hf_hub_download
                
                csv_path = hf_hub_download(
                    repo_id=self.dataset_name,
                    filename=self.csv_filename,
                    cache_dir=str(self.data_dir / "cache"),
                )
                
                processed = []
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        processed.append(self._transform(row))
                
                save_to_jsonl(processed, output_path)
        """
        pass
