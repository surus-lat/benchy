"""Mixins for tasks that cache datasets to local files.

These mixins provide automatic download-and-cache behavior using existing
dataset_utils functions. Subclasses only implement the unique transformation
logic in their _download_and_cache() method.

This eliminates ~40-60 lines of boilerplate per task.
"""

import csv
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from .utils.dataset_utils import load_jsonl_dataset, save_to_jsonl

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
        """
        Load the dataset from a local JSONL cache, creating the cache by downloading if missing.
        
        If the cached file does not exist, calls the subclass-implemented `_download_and_cache` to produce it.
        
        Returns:
            List[Dict[str, Any]]: The dataset loaded from the JSONL cache; each element is a mapping representing one example.
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
        """
        Download, transform, and persist the dataset to the given JSONL file path.
        
        Subclasses must implement this to obtain the raw data, convert it into a list of JSON-serializable records, and write the result to output_path.
        
        Parameters:
            output_path (Path): File path where the processed JSONL dataset should be written.
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
        """
        Ensure the TSV-derived dataset is cached as a JSONL file and return its records.
        
        If the cached JSONL file is missing, triggers the subclass-implemented download-and-cache routine for `self.tsv_filename` before loading.
        
        Returns:
            List[Dict[str, Any]]: Records loaded from the cached JSONL file.
        """
        file_path = self.data_dir / self.dataset_file
        
        if not file_path.exists():
            logger.info(f"Cached file not found, downloading TSV: {self.tsv_filename}")
            self._download_tsv_and_cache(file_path)
        
        return load_jsonl_dataset(file_path)
    
    @abstractmethod
    def _download_tsv_and_cache(self, output_path: Path):
        """
        Download the TSV source, transform its rows to JSON-serializable objects, and save them as a JSONL file at output_path.
        
        Subclasses must implement this to obtain the TSV (typically identified by self.tsv_filename), convert each row into the desired dict structure for the task, and persist the resulting list using save_to_jsonl to the given output_path.
        
        Parameters:
            output_path (Path): Filesystem path where the produced JSONL file should be written.
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
        """
        Load the dataset from a cached JSONL file, creating the cache by downloading and converting the CSV if the cached file is missing.
        
        If the cache file does not exist, this method calls the subclass-implemented `_download_csv_and_cache(output_path)` to download the CSV, transform rows to JSON objects, and save them as JSONL at the cache path.
        
        Returns:
            List[Dict[str, Any]]: The dataset as a list of JSON-like records loaded from the cached JSONL file.
        """
        file_path = self.data_dir / self.dataset_file
        
        if not file_path.exists():
            logger.info(f"Cached file not found, downloading CSV: {self.csv_filename}")
            self._download_csv_and_cache(file_path)
        
        return load_jsonl_dataset(file_path)
    
    @abstractmethod
    def _download_csv_and_cache(self, output_path: Path):
        """
        Download a CSV source, transform its rows into JSON-serializable records, and save them to `output_path` as JSONL.
        
        Subclasses must implement this to retrieve the CSV (e.g., from a remote repo), convert each row into the desired dictionary shape, and persist the resulting list using `save_to_jsonl(output_path)`. The method should ensure `output_path` is created or overwritten with the processed JSONL content.
        
        Parameters:
            output_path (Path): Target filesystem path where the resulting JSONL file will be written.
        """
        pass
