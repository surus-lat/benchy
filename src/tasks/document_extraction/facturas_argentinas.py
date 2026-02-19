"""Facturas Argentinas - structured invoice extraction from images."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..common import MultimodalStructuredHandler

logger = logging.getLogger(__name__)


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _coerce_record_list(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    if isinstance(raw, dict):
        # If keys are documento_ids -> record dict
        records: List[Dict[str, Any]] = []
        for key, value in raw.items():
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("documento_id", key)
                records.append(record)
        return records
    return []


def _schema_properties(schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    props = schema.get("properties")
    return props if isinstance(props, dict) else None


def _unwrap_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    if schema.get("type") != "json_schema":
        return schema
    inner = schema.get("json_schema")
    if not isinstance(inner, dict):
        return schema
    inner_schema = inner.get("schema")
    return inner_schema if isinstance(inner_schema, dict) else schema


def _filter_to_schema(value: Any, schema: Dict[str, Any]) -> Any:
    """Drop keys not defined in schema.properties (best-effort, recursive)."""
    if not isinstance(schema, dict):
        return value

    schema_type = schema.get("type")
    if schema_type == "object" and isinstance(value, dict):
        props = _schema_properties(schema)
        if not props:
            return value
        filtered: Dict[str, Any] = {}
        for key, subschema in props.items():
            if key in value:
                filtered[key] = _filter_to_schema(value[key], subschema if isinstance(subschema, dict) else {})
            else:
                # Keep missing fields as-is; prompting rules handle null/[].
                pass
        return filtered

    if schema_type == "array" and isinstance(value, list):
        items_schema = schema.get("items")
        if not isinstance(items_schema, dict):
            return value
        return [_filter_to_schema(item, items_schema) for item in value]

    return value


def _default_for_schema(schema: Dict[str, Any]) -> Any:
    schema_type = schema.get("type")
    if schema_type == "array":
        return []
    return None


def _fill_missing_fields(value: Any, schema: Dict[str, Any]) -> Any:
    """Fill missing object properties with null / [] (best-effort, recursive)."""
    if not isinstance(schema, dict):
        return value

    schema_type = schema.get("type")

    if schema_type == "object" and isinstance(value, dict):
        props = _schema_properties(schema) or {}
        filled: Dict[str, Any] = dict(value)

        for key, subschema in props.items():
            sub = subschema if isinstance(subschema, dict) else {}
            if key not in filled:
                filled[key] = _default_for_schema(sub)
                continue
            if filled[key] is None:
                continue
            filled[key] = _fill_missing_fields(filled[key], sub)

        return filled

    if schema_type == "array" and isinstance(value, list):
        items_schema = schema.get("items")
        if not isinstance(items_schema, dict):
            return value
        return [_fill_missing_fields(item, items_schema) for item in value]

    return value


class FacturasArgentinas(MultimodalStructuredHandler):
    """Extract structured data from Argentinian invoice (factura) images.

    Dataset layout (from HF snapshot or local folder):
      - schema.json
      - facturas.json
      - documento_ids.txt (optional subset, one id per line)
      - jpg/ (images)
    """

    name = "facturas_argentinas"
    display_name = "Facturas Argentinas"
    description = "Extract structured invoice data from factura images (Argentina)"

    input_type = "image"
    requires_multimodal = True
    requires_files = True
    requires_schema = True

    capability_requirements = {
        "requires_multimodal": "required",
        "requires_files": "required",
        "requires_schema": "required",
    }

    dataset_repo_id = "mauroibz/facturas-argentinas_2"

    system_prompt = "You are a precise information extraction system."
    user_prompt_template = (
        "Extract all information from the invoice image and return a JSON object\n"
        "that strictly matches the invoice schema.\n\n"
        "Rules:\n"
        "- Use null when a field is missing, [] for missing lists\n"
        "- Dates must be YYYY-MM-DD\n"
        "- Numbers must be numeric, not strings\n"
        "- Do not include additional fields\n"
        "- Output JSON only\n\n"
        "Schema:\n"
        "{schema}"
    )

    # Default scoring/tolerance tuned for invoices:
    # - Numeric correctness should dominate (totals, amounts, IDs)
    # - Strings can be slightly fuzzy (SRL vs S.R.L, punctuation, casing)
    # - CUIT/IDs are treated as numeric strings: mismatch is critical
    metrics_config = {
        "partial_credit": 0.30,
        "partial_matching": {
            "string": {
                "exact_threshold": 0.85,
                "partial_threshold": 0.40,
                "token_overlap_weight": 0.5,
                "levenshtein_weight": 0.3,
                "containment_weight": 0.2,
            },
            "number": {
                "relative_tolerance": 0.0,
                "absolute_tolerance": 0.0,
            },
        },
        "numeric_string_fields": [
            "emisor.cuit",
            "receptor.documento_numero",
            "autorizacion.codigo",
        ],
        "critical_string_fields": [
            "emisor.razon_social",
            "receptor.razon_social",
        ],
        "extraction_quality_score": {
            "weights": {
                "schema_validity": 0.15,
                "field_f1_partial": 0.70,
                "inverted_hallucination": 0.15,
            },
        },
        "document_extraction_score": {
            "weights": {
                "numeric_precision_rate": 0.50,
                "field_f1_partial": 0.25,
                "schema_validity": 0.15,
                "inverted_critical_error_rate": 0.10,
            },
        },
    }

    def _dataset_paths(self) -> Tuple[Path, Path, Path, Optional[Path]]:
        schema_path = self.data_dir / "schema.json"
        facturas_path = self.data_dir / "facturas.json"
        images_dir = self.data_dir / "jpg"
        ids_path = self.data_dir / "documento_ids.txt"
        return schema_path, facturas_path, images_dir, (ids_path if ids_path.exists() else None)

    def _load_samples(self) -> List[Dict[str, Any]]:
        schema_path, facturas_path, images_dir, ids_path = self._dataset_paths()

        # Ensure dataset is present (download/copy on first run).
        if not (schema_path.exists() and facturas_path.exists() and images_dir.exists()):
            self._copy_source_data()

        if not schema_path.exists():
            raise FileNotFoundError(f"schema.json not found in {self.data_dir}")
        if not facturas_path.exists():
            raise FileNotFoundError(f"facturas.json not found in {self.data_dir}")
        if not images_dir.exists():
            raise FileNotFoundError(f"jpg/ directory not found in {self.data_dir}")

        # (Re)load schema for prompt/validation.
        with open(schema_path, "r", encoding="utf-8") as handle:
            raw_schema = json.load(handle)
            schema = _unwrap_schema(raw_schema)
        if isinstance(raw_schema, dict) and raw_schema.get("type") == "json_schema":
            logger.info("Unwrapped OpenAI response_format schema to a plain JSON Schema for prompting/validation")
        self.schema = schema

        selected_ids: Optional[Set[str]] = None
        if ids_path:
            selected_ids = set(_read_lines(ids_path))
            logger.info(f"Loaded {len(selected_ids)} documento_ids from {ids_path.name}")

        # Index images by stem for fast lookup.
        image_index: Dict[str, Path] = {}
        for image_path in images_dir.iterdir():
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            image_index.setdefault(image_path.stem, image_path)

        with open(facturas_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        records = _coerce_record_list(raw)

        dataset: List[Dict[str, Any]] = []
        for idx, record in enumerate(records):
            documento_id = record.get("documento_id") or record.get("id") or record.get("documentoId")
            documento_id = str(documento_id) if documento_id is not None else ""

            if selected_ids is not None and documento_id and documento_id not in selected_ids:
                continue

            # Resolve image path.
            image_path = None
            for key in ("image", "image_path", "filename", "file_name"):
                candidate = record.get(key)
                if isinstance(candidate, str) and candidate:
                    candidate_path = Path(candidate)
                    if not candidate_path.is_absolute():
                        candidate_path = images_dir / candidate_path.name
                    if candidate_path.exists():
                        image_path = candidate_path
                        break

            if image_path is None and documento_id:
                image_path = image_index.get(documento_id)

            if image_path is None:
                logger.warning(f"Skipping sample {idx}: image not found for documento_id={documento_id!r}")
                continue

            expected = _fill_missing_fields(_filter_to_schema(record, schema), schema)

            dataset.append(
                {
                    "id": f"{self.name}_{idx:06d}_{documento_id or image_path.stem}",
                    "image_path": str(image_path),
                    "schema": schema,
                    "expected": expected,
                    "documento_id": documento_id or image_path.stem,
                }
            )

        logger.info(f"Loaded {len(dataset)} samples")
        return dataset
