"""Chat Structured Extraction subtask using handler system."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any

from ..common import StructuredHandler, CachedDatasetMixin, download_huggingface_dataset, save_to_jsonl

logger = logging.getLogger(__name__)


class ChatExtract(CachedDatasetMixin, StructuredHandler):
    """Chat structured extraction task."""

    # Task configuration
    name = "chat_extract"
    display_name = "Chat Structured Extraction"
    description = "Extract structured data from chat conversations"
    
    dataset_name = "mauroibz/chat_structured_extraction"
    split = "train"
    dataset_file = "chat_extract_data.jsonl"
    
    # Prompts
    system_prompt = "You are a precise data extraction assistant. Extract information from the provided text according to the given JSON schema. Only extract information explicitly stated in the text. If information for a field is not present, use null as appropriate."
    
    # Metrics configuration
    metrics_config = {
        "extraction_quality_score": {
            "enabled": True,
            "weights": {
                "schema_validity": 0.2,
                "field_f1_partial": 0.6,
                "inverted_hallucination": 0.2,
            },
        },
        "partial_matching": {
            "string": {
                "token_overlap_weight": 0.5,
                "levenshtein_weight": 0.3,
                "containment_weight": 0.2,
                "exact_threshold": 0.95,
                "partial_threshold": 0.5,
            },
            "number": {
                "relative_tolerance": 0.001,
                "absolute_tolerance": 1e-06,
            },
            "array": {
                "method": "jaccard",
                "partial_credit": True,
            },
        },
        "normalization": {
            "case_sensitive": False,
            "normalize_whitespace": True,
            "unicode_normalize": True,
        },
    }

    def _download_and_cache(self, output_path: Path):
        """Download and preprocess chat extraction dataset."""
        raw_samples = download_huggingface_dataset(
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=str(self.data_dir / "cache"),
        )
        
        # Load or define schema
        schema_path = self.data_dir / "schema_expected_lead_data.json"
        if schema_path.exists():
            with open(schema_path, "r") as f:
                schema = json.load(f)
        else:
            # Define the lead data schema if file doesn't exist
            schema = {
                "type": "object",
                "properties": {
                    "nombre": {"type": "string"},
                    "tiene_negocio": {"type": "boolean"},
                    "negocio": {
                        "type": "object",
                        "properties": {
                            "descripcion_negocio": {"type": "string"},
                            "meses_en_negocio": {"type": "integer"},
                            "cantidad_empleados": {"type": "integer"}
                        },
                        "additionalProperties": False
                    }
                },
                "additionalProperties": False
            }
        
        processed = []
        for idx, raw_sample in enumerate(raw_samples):
            # Extract user messages from input field
            input_data = raw_sample.get("input", {})
            if isinstance(input_data, dict):
                user_messages = input_data.get("user_messages", [])
                # Join user messages into a conversation
                text = "\n".join(user_messages) if isinstance(user_messages, list) else str(input_data)
            else:
                text = str(input_data)
            
            # Extract expected output from output field
            output_data = raw_sample.get("output", {})
            if isinstance(output_data, dict):
                expected = output_data.get("expected_lead_data", output_data)
            else:
                expected = {}
            
            # Truncate if too long
            if len(text) > 20000:
                text = text[:20000]
            
            processed.append({
                "id": raw_sample.get("id", f"{idx:06d}"),
                "text": text,
                "schema": schema,
                "expected": expected,
            })
        
        save_to_jsonl(processed, output_path)

    def preprocess_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """Samples are already preprocessed in _download_and_cache, return as-is."""
        return raw_sample
    
    def get_prompt(self, sample: Dict[str, Any]) -> tuple[str, str]:
        """Build prompts for structured extraction."""
        schema = sample.get("schema", {})
        schema_str = json.dumps(schema, indent=2) if schema else ""

        user_prompt = (
            f"Text:\n{sample.get('text', '')}\n\n"
            f"Extract information according to this JSON schema:\n"
            f"{schema_str}\n\n"
            f"Output valid JSON matching the schema exactly."
        )

        return self.system_prompt, user_prompt
