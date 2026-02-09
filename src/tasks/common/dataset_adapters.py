"""Dataset adapter for unified dataset loading from multiple sources.

This module provides a unified interface for loading datasets from:
- HuggingFace datasets
- Local JSONL files
- Local directory structures (for multimodal tasks)

The adapter handles field mapping, normalization, and caching automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DatasetAdapter:
    """Unified dataset loader supporting multiple sources.
    
    This adapter consolidates all dataset loading logic, replacing scattered
    implementations across tasks. It provides:
    - HuggingFace dataset loading with auto-caching
    - Local JSONL file loading
    - Local directory structure loading (for multimodal)
    - Field mapping and normalization
    - Smart source detection
    
    Example:
        adapter = DatasetAdapter()
        samples = adapter.load(
            dataset_config={
                "name": "climatebert/environmental_claims",
                "source": "auto",
                "split": "test",
                "input_field": "text",
                "output_field": "label",
            },
            cache_dir=Path(".data")
        )
    """
    
    def load(
        self,
        dataset_config: Dict[str, Any],
        cache_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Load dataset from configured source.
        
        Args:
            dataset_config: Dataset configuration with keys:
                - name: Dataset identifier (HF path, local path, or directory)
                - source: "auto", "huggingface", "local", "directory" (default: "auto")
                - split: Dataset split for HF datasets (default: "test")
                - input_field: Field name for input text (default: "text")
                - output_field: Field name for expected output (default: "expected")
                - id_field: Field name for sample ID (default: "id", auto-generated if missing)
                - schema_field: Field name for schema (structured tasks)
                - label_field: Field name for labels (classification)
                - choices_field: Field name for choices (classification)
                - multimodal_input: Whether dataset has multimodal inputs
                - multimodal_image_field: Field name for image paths (default: "image_path")
            cache_dir: Directory for caching downloaded datasets
            
        Returns:
            List of normalized sample dicts with standard fields:
                - id: Sample identifier
                - text: Input text (or image path for multimodal)
                - expected: Expected output
                - (task-specific fields: schema, choices, label, etc.)
                
        Raises:
            ValueError: If dataset configuration is invalid
            FileNotFoundError: If local dataset not found
        """
        name = dataset_config.get("name")
        if not name:
            raise ValueError("Dataset configuration must include 'name'")
        
        source = dataset_config.get("source", "auto")
        
        # Auto-detect source if not specified
        if source == "auto":
            source = self._detect_source(name)
            logger.info(f"Auto-detected dataset source: {source}")
        
        # Load from appropriate source
        if source == "huggingface":
            return self._load_huggingface(dataset_config, cache_dir)
        elif source == "local":
            return self._load_jsonl(dataset_config)
        elif source == "directory":
            return self._load_directory(dataset_config)
        else:
            raise ValueError(
                f"Invalid dataset source: {source}. "
                f"Must be 'auto', 'huggingface', 'local', or 'directory'"
            )
    
    def _detect_source(self, name: str) -> str:
        """Auto-detect dataset source from name.
        
        Args:
            name: Dataset name/path
            
        Returns:
            Detected source: "huggingface", "local", or "directory"
        """
        path = Path(name)
        
        # Check if it's a local path
        if path.exists():
            if path.is_dir():
                return "directory"
            elif path.suffix == ".jsonl":
                return "local"
        
        # If it looks like a HuggingFace dataset (org/dataset), try HF
        if "/" in name and not path.exists():
            return "huggingface"
        
        # Default to HuggingFace (will try and fail with clear error)
        return "huggingface"
    
    def _load_huggingface(
        self,
        dataset_config: Dict[str, Any],
        cache_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Load dataset from HuggingFace Hub.
        
        Args:
            dataset_config: Dataset configuration
            cache_dir: Cache directory for downloaded data
            
        Returns:
            List of normalized samples
        """
        from .utils.dataset_utils import download_huggingface_dataset, save_to_jsonl
        
        name = dataset_config["name"]
        split = dataset_config.get("split", "test")
        
        # Create cache directory
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate cache filename based on dataset name and split
        cache_filename = f"{name.replace('/', '_')}_{split}.jsonl"
        cache_path = cache_dir / cache_filename
        
        # Check if cached
        if cache_path.exists():
            logger.info(f"Loading cached dataset from {cache_path}")
            return self._load_jsonl({"name": str(cache_path), **dataset_config})
        
        # Download from HuggingFace
        logger.info(f"Downloading dataset {name} (split: {split}) from HuggingFace")
        try:
            raw_dataset = download_huggingface_dataset(
                dataset_name=name,
                split=split,
                cache_dir=str(cache_dir / "hf_cache"),
            )
        except Exception as e:
            raise ValueError(
                f"Failed to download HuggingFace dataset '{name}': {e}\n"
                f"Make sure the dataset exists and you have access."
            ) from e
        
        # Normalize and apply field mappings
        samples = []
        for idx, raw_sample in enumerate(raw_dataset):
            sample = self._normalize_sample(raw_sample, dataset_config, idx)
            if sample is not None:
                samples.append(sample)
        
        # Cache for future use
        save_to_jsonl(samples, cache_path)
        logger.info(f"Cached {len(samples)} samples to {cache_path}")
        
        return samples
    
    def _load_jsonl(self, dataset_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load dataset from local JSONL file.
        
        Args:
            dataset_config: Dataset configuration
            
        Returns:
            List of normalized samples
        """
        from .utils.dataset_utils import load_jsonl_dataset
        
        path = Path(dataset_config["name"])
        
        if not path.exists():
            raise FileNotFoundError(
                f"JSONL dataset not found: {path}\n"
                f"Make sure the file exists and the path is correct."
            )
        
        logger.info(f"Loading dataset from {path}")
        raw_samples = load_jsonl_dataset(path)
        
        # Normalize and apply field mappings
        samples = []
        for idx, raw_sample in enumerate(raw_samples):
            sample = self._normalize_sample(raw_sample, dataset_config, idx)
            if sample is not None:
                samples.append(sample)
        
        logger.info(f"Loaded {len(samples)} samples from {path}")
        return samples
    
    def _load_directory(self, dataset_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load dataset from directory structure.
        
        This is primarily for multimodal tasks where images/files are organized
        in directories. The structure depends on the task type.
        
        Args:
            dataset_config: Dataset configuration
            
        Returns:
            List of normalized samples
        """
        path = Path(dataset_config["name"])
        
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(
                f"Directory not found: {path}\n"
                f"Make sure the directory exists and the path is correct."
            )
        
        logger.info(f"Loading dataset from directory: {path}")
        
        # For directory loading, we expect the task to handle the structure
        # This is a basic implementation that looks for common patterns
        samples = []
        
        # Check for JSONL metadata file
        metadata_file = path / "metadata.jsonl"
        if metadata_file.exists():
            logger.info(f"Found metadata file: {metadata_file}")
            # Keep dataset options but force JSONL loader to read metadata file.
            raw_samples = self._load_jsonl({**dataset_config, "name": str(metadata_file)})
            
            # Resolve relative paths for multimodal fields
            if dataset_config.get("multimodal_input"):
                image_field = dataset_config.get("multimodal_image_field", "image_path")
                for sample in raw_samples:
                    if image_field in sample:
                        image_path = Path(sample[image_field])
                        if not image_path.is_absolute():
                            sample[image_field] = str(path / image_path)
            
            return raw_samples
        
        # Otherwise, scan for image files (basic implementation)
        if dataset_config.get("multimodal_input"):
            image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
            image_files = [
                f for f in path.rglob("*")
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            
            logger.info(f"Found {len(image_files)} image files")
            
            for idx, image_file in enumerate(sorted(image_files)):
                sample = {
                    "id": f"dir_{idx:06d}",
                    "text": str(image_file),  # For multimodal, text is the image path
                    "image_path": str(image_file),
                }
                samples.append(sample)
            
            return samples
        
        raise ValueError(
            f"Directory dataset loading requires either:\n"
            f"1. A metadata.jsonl file in the directory, or\n"
            f"2. multimodal_input=True for automatic image scanning\n"
            f"Directory: {path}"
        )
    
    def _normalize_sample(
        self,
        raw_sample: Dict[str, Any],
        dataset_config: Dict[str, Any],
        idx: int,
    ) -> Optional[Dict[str, Any]]:
        """Normalize a raw sample to standard format.
        
        Args:
            raw_sample: Raw sample from dataset
            dataset_config: Dataset configuration with field mappings
            idx: Sample index (for auto-generating IDs)
            
        Returns:
            Normalized sample dict or None if sample should be skipped
        """
        # If sample is already normalized (has id, text, expected), return as-is
        # This allows tasks to pre-process samples in their own way
        if "id" in raw_sample and "text" in raw_sample:
            return raw_sample
        
        # Extract field mappings from config
        id_field = dataset_config.get("id_field", "id")
        input_field = dataset_config.get("input_field", "text")
        output_field = dataset_config.get("output_field", "expected")
        
        # Build normalized sample
        sample = {}
        
        # ID (auto-generate if missing)
        if id_field in raw_sample:
            sample["id"] = str(raw_sample[id_field])
        else:
            sample["id"] = f"sample_{idx:06d}"
        
        # Input text
        if input_field in raw_sample:
            sample["text"] = raw_sample[input_field]
        else:
            logger.warning(
                f"Sample {sample['id']}: Missing input field '{input_field}', skipping"
            )
            return None
        
        # Expected output
        if output_field in raw_sample:
            sample["expected"] = raw_sample[output_field]
        else:
            # For some datasets, expected might be optional (e.g., inference-only)
            logger.debug(f"Sample {sample['id']}: Missing output field '{output_field}'")
        
        # Task-specific fields (pass through if present)
        task_fields = [
            "schema",
            "schema_field",
            "label",
            "label_field",
            "choices",
            "choice_labels",
            "label_to_index",
            "label_value",
            "image_path",
        ]
        
        for field in task_fields:
            # Check both the field name and the configured field name
            config_field = dataset_config.get(f"{field}_field", field)
            if config_field in raw_sample:
                sample[field] = raw_sample[config_field]
            elif field in raw_sample:
                sample[field] = raw_sample[field]
        
        # Copy any other fields that might be useful
        for key, value in raw_sample.items():
            if key not in sample and key not in [id_field, input_field, output_field]:
                sample[key] = value
        
        return sample


def validate_dataset_config(dataset_config: Dict[str, Any]) -> List[str]:
    """Validate dataset configuration.
    
    Args:
        dataset_config: Dataset configuration to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if not dataset_config.get("name"):
        errors.append("Dataset configuration must include 'name'")
    
    source = dataset_config.get("source", "auto")
    valid_sources = ["auto", "huggingface", "local", "directory"]
    if source not in valid_sources:
        errors.append(
            f"Invalid source '{source}'. Must be one of: {', '.join(valid_sources)}"
        )
    
    return errors
