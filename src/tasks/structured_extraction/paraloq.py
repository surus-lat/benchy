"""Paraloq JSON Data Extraction subtask using handler system."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List

from ..common import StructuredHandler, CachedDatasetMixin, download_huggingface_dataset, save_to_jsonl

logger = logging.getLogger(__name__)


class Paraloq(CachedDatasetMixin, StructuredHandler):
    """Paraloq structured extraction task."""

    # Task configuration
    name = "paraloq"
    display_name = "Paraloq JSON Extraction"
    description = "Extract structured data from diverse document types"
    
    dataset_name = "paraloq/json_data_extraction"
    split = "train"
    dataset_file = "paraloq_data.jsonl"
    
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
        """
        Download the Paraloq dataset, preprocess each sample, and write processed records to output_path.
        
        Parses the dataset's `schema` and `item` fields from JSON strings when necessary, truncates `text` to 20,000 characters if longer, and skips samples whose `schema` or `item` cannot be decoded as JSON. Writes a JSONL file where each record contains `id` (zero-padded six-digit index), `text`, `schema`, and `expected`.
        
        Parameters:
            output_path (Path): File path to write the processed JSONL dataset.
        """
        raw_samples = download_huggingface_dataset(
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=str(self.data_dir / "cache"),
        )
        
        processed = []
        for idx, raw_sample in enumerate(raw_samples):
            text = raw_sample.get("text", "")
            
            # Parse schema from string to dict
            schema_raw = raw_sample.get("schema", "{}")
            if isinstance(schema_raw, str):
                try:
                    schema = json.loads(schema_raw)
                except json.JSONDecodeError:
                    logger.warning(f"Sample {idx}: Failed to parse schema, skipping")
                    continue
            else:
                schema = schema_raw
            
            # Expected output is in 'item' field (also a string that needs parsing)
            item_raw = raw_sample.get("item", "{}")
            if isinstance(item_raw, str):
                try:
                    expected = json.loads(item_raw)
                except json.JSONDecodeError:
                    logger.warning(f"Sample {idx}: Failed to parse expected output, skipping")
                    continue
            else:
                expected = item_raw
            
            # Truncate if too long
            if len(text) > 20000:
                text = text[:20000]
            
            processed.append({
                "id": f"{idx:06d}",
                "text": text,
                "schema": schema,
                "expected": expected,
            })
        
        save_to_jsonl(processed, output_path)

    def preprocess_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """
        Return the input sample unchanged.
        
        Parameters:
            raw_sample (Dict[str, Any]): The raw dataset sample.
            idx (int): The sample's index in the dataset.
        
        Returns:
            Dict[str, Any]: The same sample dictionary that was provided.
        """
        return raw_sample
    
    def get_prompt(self, sample: Dict[str, Any]) -> tuple[str, str]:
        """
        Constructs the system and user prompts for extracting JSON-structured data from a sample.
        
        The user prompt includes the sample text, a pretty-printed JSON schema (if present), and an explicit instruction to output valid JSON that matches the schema exactly.
        
        Parameters:
            sample (Dict[str, Any]): Input sample expected to contain:
                - "text": the source text to extract from.
                - "schema": a JSON-serializable mapping describing the expected output schema.
        
        Returns:
            Tuple[str, str]: A tuple (system_prompt, user_prompt) where `system_prompt` is the handler's predefined system instruction and `user_prompt` contains the text, schema, and extraction directive.
        """
        schema = sample.get("schema", {})
        schema_str = json.dumps(schema, indent=2) if schema else ""

        user_prompt = (
            f"Text:\n{sample.get('text', '')}\n\n"
            f"Extract information according to this JSON schema:\n"
            f"{schema_str}\n\n"
            f"Output valid JSON matching the schema exactly."
        )

        return self.system_prompt, user_prompt