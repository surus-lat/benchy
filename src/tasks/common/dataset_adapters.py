"""Dataset adapter for unified dataset loading from multiple sources.

This module provides a unified interface for loading datasets from:
- HuggingFace datasets
- Local JSONL files
- Local Parquet files or directories containing Parquet files
- Local directory structures (for multimodal tasks)
- Auto-discovered datasets from the .data/ directory

The adapter handles field mapping, normalization, binary materialization,
and caching automatically.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _repo_data_dir() -> Path:
    """Return the .data/ directory at the repository root."""
    return Path(__file__).resolve().parent.parent.parent.parent / ".data"


def _convert_custom_schema(
    raw_schema: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, str], Dict[str, str]]:
    """Convert a custom benchy schema (with ``fields`` array) to JSON Schema.

    Also extracts the ground-truth column mapping and field type mapping so the
    adapter can build the ``expected`` dict from dataset columns and coerce GT
    values to the correct Python type.

    Returns:
        (json_schema, gt_mapping, type_mapping) where *gt_mapping* maps extraction
        field names to dataset column names and *type_mapping* maps field names to
        their declared schema type (e.g. ``{"items": "array"}``).
    """
    properties: Dict[str, Any] = {}
    required: List[str] = []
    gt_mapping: Dict[str, str] = {}
    type_mapping: Dict[str, str] = {}

    for field in raw_schema.get("fields", []):
        fname = field["name"]
        field_type = field.get("type", "string")
        prop: Dict[str, Any] = {"type": field_type}
        if "description" in field:
            prop["description"] = field["description"]
        properties[fname] = prop

        type_mapping[fname] = field_type

        if field.get("required", False):
            required.append(fname)

        gt_col = field.get("ground_truth_field")
        if gt_col and field.get("ground_truth_available", True):
            gt_mapping[fname] = gt_col

    json_schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        json_schema["required"] = required

    return json_schema, gt_mapping, type_mapping


def _extract_gt_mapping_from_json_schema(
    schema: Dict[str, Any],
    _prefix: str = "",
) -> tuple[Dict[str, str], Dict[str, str]]:
    """Walk a standard JSON Schema and extract ``gt_field`` annotations.

    Returns:
        (gt_mapping, type_mapping) where *gt_mapping* is a flat mapping of
        dot-notation field path → dataset column name and *type_mapping* maps
        each field path to its declared schema type (e.g. ``"array"``).
    Also strips ``gt_field``, ``ground_truth_available``, ``evaluation_note``,
    and ``db_source`` keys from the schema in-place so the resulting object is
    a clean JSON Schema suitable for sending to an LLM.
    """
    gt_mapping: Dict[str, str] = {}
    type_mapping: Dict[str, str] = {}
    _annotation_keys = {"gt_field", "ground_truth_available", "evaluation_note", "db_source"}

    props = schema.get("properties")
    if not isinstance(props, dict):
        return gt_mapping, type_mapping

    for name, prop in props.items():
        path = f"{_prefix}.{name}" if _prefix else name

        # Collect field type
        type_mapping[path] = prop.get("type", "string")

        # Collect GT annotation
        gt_col = prop.get("gt_field")
        if gt_col and prop.get("ground_truth_available", False):
            gt_mapping[path] = gt_col

        # Strip annotation keys
        for k in _annotation_keys:
            prop.pop(k, None)

        # Recurse into nested objects
        if prop.get("type") == "object":
            sub_gt, sub_types = _extract_gt_mapping_from_json_schema(prop, _prefix=path)
            gt_mapping.update(sub_gt)
            type_mapping.update(sub_types)

        # Recurse into array items
        items = prop.get("items")
        if isinstance(items, dict) and items.get("type") == "object":
            sub_gt, sub_types = _extract_gt_mapping_from_json_schema(items, _prefix=path)
            gt_mapping.update(sub_gt)
            type_mapping.update(sub_types)

    # Strip annotation keys at schema root level too
    for k in _annotation_keys:
        schema.pop(k, None)

    return gt_mapping, type_mapping


def resolve_dataset_name(name: str) -> tuple[str, Dict[str, Any]]:
    """Resolve a dataset name, checking .data/ for auto-discoverable datasets.

    If *name* matches a subdirectory under ``.data/``, the directory is
    inspected for ``dataset_info.json`` and ``schema.json``.  Metadata from
    those files is returned so the caller can fill in defaults (labels,
    schema, field mappings) without requiring extra CLI flags.

    Returns:
        (resolved_name, extra_config) — *resolved_name* is the original name
        when it's not a .data/ shortcut, or the full path to the parquet
        directory when it is.  *extra_config* contains keys that can be
        merged into the dataset config (e.g. ``labels``, ``schema_path``).
    """
    data_dir = _repo_data_dir() / name
    if not data_dir.is_dir():
        return name, {}

    extra: Dict[str, Any] = {}

    # Read dataset_info.json
    info_path = data_dir / "dataset_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as fh:
            info = json.load(fh)

        # Infer label mapping from label_distribution
        label_dist = info.get("label_distribution")
        if label_dist:
            # Map each label value to itself as display text so that
            # the handler's label_to_index matches dataset column values.
            labels = {lbl: lbl for lbl in sorted(label_dist.keys())}
            extra["_inferred_labels"] = labels
            extra["_label_distribution"] = label_dist

        # Expose features for field-mapping hints
        features = info.get("features", {})
        if features:
            extra["_features"] = features

            # Detect binary columns → flag as multimodal document dataset
            binary_dtypes = {"large_binary", "binary"}
            for feat_name, feat_info in features.items():
                if isinstance(feat_info, dict) and feat_info.get("dtype") in binary_dtypes:
                    extra["_has_binary"] = True
                    extra["_binary_field"] = feat_name
                    break

        extra["_dataset_info"] = info

    # Auto-attach schema.json and extract ground-truth mappings
    schema_path = data_dir / "schema.json"
    if schema_path.exists():
        extra["schema_path"] = str(schema_path)

        with open(schema_path, "r", encoding="utf-8") as fh:
            raw_schema = json.load(fh)

        # Convert custom schema format (fields array) to JSON Schema + GT mapping
        if "fields" in raw_schema and isinstance(raw_schema["fields"], list):
            json_schema, gt_mapping, type_mapping = _convert_custom_schema(raw_schema)
            extra["_json_schema"] = json_schema
            if gt_mapping:
                extra["_ground_truth_mapping"] = gt_mapping
            if type_mapping:
                extra["_field_type_mapping"] = type_mapping
        elif "properties" in raw_schema:
            # Standard JSON Schema — extract gt_field annotations in-place
            import copy
            clean_schema = copy.deepcopy(raw_schema)
            gt_mapping, type_mapping = _extract_gt_mapping_from_json_schema(clean_schema)
            extra["_json_schema"] = clean_schema
            if gt_mapping:
                extra["_ground_truth_mapping"] = gt_mapping
            if type_mapping:
                extra["_field_type_mapping"] = type_mapping

    # Resolve to the parquet data directory or the dataset root
    parquet_subdir = data_dir / "data"
    if parquet_subdir.is_dir() and any(parquet_subdir.glob("*.parquet")):
        extra["source"] = "parquet"
        return str(parquet_subdir), extra

    # Fallback: the dataset dir itself may contain parquet files
    if any(data_dir.glob("*.parquet")):
        extra["source"] = "parquet"
        return str(data_dir), extra

    return str(data_dir), extra


def list_data_datasets() -> List[Dict[str, Any]]:
    """Scan the .data/ directory and return metadata for each dataset found.

    Returns a list of dicts with keys: name, path, description, splits,
    label_distribution, has_schema, features (column names).
    """
    data_root = _repo_data_dir()
    if not data_root.is_dir():
        return []

    datasets = []
    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        record: Dict[str, Any] = {"name": entry.name, "path": str(entry)}

        info_path = entry / "dataset_info.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as fh:
                info = json.load(fh)
            record["description"] = info.get("description", "")
            record["splits"] = info.get("splits", {})
            record["label_distribution"] = info.get("label_distribution")
            record["features"] = list((info.get("features") or {}).keys())
        else:
            record["description"] = ""
            record["splits"] = {}
            record["features"] = []

        record["has_schema"] = (entry / "schema.json").exists()

        # Detect parquet
        parquet_dir = entry / "data"
        if parquet_dir.is_dir():
            record["parquet_files"] = [f.name for f in parquet_dir.glob("*.parquet")]
        else:
            record["parquet_files"] = [f.name for f in entry.glob("*.parquet")]

        datasets.append(record)

    return datasets


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
    
    @staticmethod
    def _field_mapping_fingerprint(dataset_config: Dict[str, Any]) -> str:
        """Build a stable fingerprint for field mapping-related options."""
        mapping_config: Dict[str, Any] = {}

        for key, value in dataset_config.items():
            if key.endswith("_field") or key == "field_mapping":
                mapping_config[key] = value

        payload = json.dumps(mapping_config, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]

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
        elif source == "parquet":
            return self._load_parquet(dataset_config, cache_dir)
        elif source == "directory":
            return self._load_directory(dataset_config)
        else:
            raise ValueError(
                f"Invalid dataset source: {source}. "
                f"Must be 'auto', 'huggingface', 'local', 'parquet', or 'directory'"
            )
    
    def _detect_source(self, name: str) -> str:
        """Auto-detect dataset source from name.

        Args:
            name: Dataset name/path

        Returns:
            Detected source: "huggingface", "local", "parquet", or "directory"
        """
        path = Path(name)

        # Check if it's a local path
        if path.exists():
            if path.is_file():
                if path.suffix == ".parquet":
                    return "parquet"
                if path.suffix == ".jsonl":
                    return "local"
            if path.is_dir():
                # Directory containing parquet files → parquet source
                if any(path.glob("*.parquet")):
                    return "parquet"
                return "directory"

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
        mapping_fingerprint = self._field_mapping_fingerprint(dataset_config)
        
        # Create cache directory
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate cache filename based on dataset name, split, and field mapping.
        cache_filename = f"{name.replace('/', '_')}_{split}_{mapping_fingerprint}.jsonl"
        cache_path = cache_dir / cache_filename
        
        # Check if cached
        if cache_path.exists():
            logger.info(f"Loading cached dataset from {cache_path}")
            return self._load_jsonl({**dataset_config, "name": str(cache_path)})
        
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
    
    # ------------------------------------------------------------------
    # Parquet loading + binary materialization
    # ------------------------------------------------------------------

    def _load_parquet(
        self,
        dataset_config: Dict[str, Any],
        cache_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Load dataset from local Parquet file(s).

        Supports a single ``.parquet`` file or a directory containing one or
        more Parquet files (one per split).  When the config specifies a
        ``split`` (default ``"test"``), a file named ``<split>.parquet`` is
        preferred if the path is a directory.

        Binary columns (``large_binary`` / ``binary``) are materialised to
        disk automatically so downstream interfaces receive file paths.
        """
        import pyarrow.parquet as pq

        name = dataset_config["name"]
        split = dataset_config.get("split", "test")
        path = Path(name)

        if not path.exists():
            raise FileNotFoundError(f"Parquet path not found: {path}")

        # Determine which file to read
        if path.is_file():
            parquet_path = path
        else:
            # Directory — pick split file or fall back to first parquet
            split_file = path / f"{split}.parquet"
            if split_file.exists():
                parquet_path = split_file
            else:
                parquet_files = sorted(path.glob("*.parquet"))
                if not parquet_files:
                    raise FileNotFoundError(
                        f"No .parquet files found in {path}"
                    )
                parquet_path = parquet_files[0]
                logger.warning(
                    "Split '%s' not found, falling back to %s",
                    split, parquet_path.name,
                )

        logger.info("Loading parquet dataset from %s", parquet_path)
        table = pq.read_table(parquet_path)

        # Detect binary columns for materialisation
        binary_col = dataset_config.get("binary_field", None)
        ext_col = dataset_config.get("binary_extension_field", None)

        if binary_col is None:
            # Auto-detect: look for large_binary / binary typed columns
            import pyarrow as pa
            for field in table.schema:
                if pa.types.is_large_binary(field.type) or pa.types.is_binary(field.type):
                    binary_col = field.name
                    break

        if binary_col and ext_col is None:
            # Try common extension column names
            for candidate in ("input_file_extension", "file_extension", "extension", "ext"):
                if candidate in table.column_names:
                    ext_col = candidate
                    break

        # Determine materialisation cache directory
        # Place it next to the parquet file (sibling "cache" dir)
        mat_dir: Optional[Path] = None
        if binary_col and binary_col in table.column_names:
            mat_dir = parquet_path.parent.parent / "cache"
            mat_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Will materialise binary column '%s' to %s", binary_col, mat_dir
            )

        # Convert to Python dicts, materialising binary columns
        samples = []
        id_field = dataset_config.get("id_field", "record_id")
        # Fallback id fields common in these datasets
        id_candidates = [id_field, "record_id", "id", "attachment_id"]

        for i in range(table.num_rows):
            row: Dict[str, Any] = {}
            for col_name in table.column_names:
                value = table.column(col_name)[i].as_py()
                if col_name == binary_col and mat_dir is not None:
                    # Materialise binary blob to disk
                    row_id = self._row_id(table, i, id_candidates)
                    ext = "bin"
                    if ext_col and ext_col in table.column_names:
                        ext = (table.column(ext_col)[i].as_py() or "bin").lstrip(".")
                    file_path = mat_dir / f"{row_id}.{ext}"
                    if not file_path.exists() and value is not None:
                        file_path.write_bytes(value)

                    # Optionally render non-image documents (PDF, TIFF, …) to PNG.
                    # Enabled by render_documents=True (auto-set for LLM providers).
                    # Custom APIs that accept raw files should leave this off.
                    render = dataset_config.get("render_documents", False)
                    if render:
                        from src.interfaces.common.image_preprocessing import needs_rendering, render_document_to_image
                        if needs_rendering(str(file_path)):
                            try:
                                rendered = render_document_to_image(
                                    str(file_path),
                                    dpi=dataset_config.get("render_dpi", 200),
                                    max_pages=dataset_config.get("render_max_pages", 1),
                                )
                                row["file_path"] = rendered
                            except (ImportError, ValueError) as exc:
                                logger.warning(
                                    "Could not render %s to image: %s. "
                                    "Install pymupdf for PDF support.",
                                    file_path.name, exc,
                                )
                                row["file_path"] = str(file_path)
                        else:
                            row["file_path"] = str(file_path)
                    else:
                        row["file_path"] = str(file_path)
                    # Don't store the raw bytes in the sample dict
                    continue
                row[col_name] = value

            sample = self._normalize_sample(row, dataset_config, i)
            if sample is not None:
                samples.append(sample)

        logger.info("Loaded %d samples from %s", len(samples), parquet_path)
        return samples

    @staticmethod
    def _row_id(table, row_idx: int, id_candidates: List[str]) -> str:
        """Extract a stable identifier for a row, trying several column names."""
        for col in id_candidates:
            if col in table.column_names:
                val = table.column(col)[row_idx].as_py()
                if val is not None:
                    return str(val)
        return f"row_{row_idx:06d}"

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
        output_field = dataset_config.get("output_field", "expected")
        if (
            "id" in raw_sample
            and "text" in raw_sample
            and ("expected" in raw_sample or output_field in raw_sample)
        ):
            return raw_sample
        
        # Extract field mappings from config
        id_field = dataset_config.get("id_field", "id")
        input_field = dataset_config.get("input_field", "text")
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
        elif "file_path" in raw_sample:
            # For multimodal / document datasets the materialised file IS the
            # input.  Use file_path as text so downstream interfaces can pick
            # it up (the prompt template or interface will use the path).
            sample["text"] = raw_sample["file_path"]
        else:
            logger.warning(
                f"Sample {sample['id']}: Missing input field '{input_field}', skipping"
            )
            return None
        
        # Expected output
        gt_mapping = dataset_config.get("_ground_truth_mapping")
        if gt_mapping:
            # Build expected dict from ground-truth columns (extraction datasets).
            # Dot-notation keys (e.g. "encabezado.ugl_dependencia") become nested dicts.
            field_type_mapping = dataset_config.get("_field_type_mapping", {})
            expected: Dict[str, Any] = {}
            for field_name, gt_col in gt_mapping.items():
                if gt_col not in raw_sample or raw_sample[gt_col] is None:
                    continue
                value = raw_sample[gt_col]
                # When the schema declares the field as an array but the parquet
                # column stores a JSON-serialised string, parse it before comparison
                # so the comparison logic sees a list on both sides.
                if field_type_mapping.get(field_name) == "array" and isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            value = parsed
                        else:
                            logger.warning(
                                "GT field %r: json.loads returned %s, expected list — "
                                "treating as GT data error, skipping field",
                                field_name, type(parsed).__name__,
                            )
                            continue
                    except json.JSONDecodeError:
                        logger.warning(
                            "GT field %r: string value is not valid JSON — "
                            "treating as GT data error, skipping field",
                            field_name,
                        )
                        continue
                parts = field_name.split(".")
                target = expected
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value
            if expected:
                sample["expected"] = expected
        elif output_field in raw_sample:
            sample["expected"] = raw_sample[output_field]
        else:
            # For some datasets, expected might be optional (e.g., inference-only)
            logger.debug(f"Sample {sample['id']}: Missing output field '{output_field}'")
        
        # Propagate materialised file_path for multimodal workflows.
        if "file_path" in raw_sample and "file_path" not in sample:
            sample["file_path"] = raw_sample["file_path"]

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
    valid_sources = ["auto", "huggingface", "local", "parquet", "directory"]
    if source not in valid_sources:
        errors.append(
            f"Invalid source '{source}'. Must be one of: {', '.join(valid_sources)}"
        )
    
    return errors
