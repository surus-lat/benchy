"""Paraloq JSON Data Extraction subtask using handler system.

This subtask evaluates models on extracting structured information from unstructured
text across diverse domains including medical records, e-commerce listings, business
documents, travel itineraries, media content, and technology specifications.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Any

from ..formats import StructuredHandler

logger = logging.getLogger(__name__)

# Data and cache directories relative to this module
DATA_DIR = Path(__file__).parent.parent / 'structured' / '.data'
CACHE_DIR = Path(__file__).parent.parent / 'structured' / 'cache'


class Paraloq(StructuredHandler):
    """Paraloq structured extraction task.
    
    This task uses preprocessed JSONL data from the paraloq/json_data_extraction
    dataset. The data includes diverse document types with associated JSON schemas
    for extraction.
    """

    # Task configuration
    name = "paraloq"
    dataset = None  # Use local JSONL file
    default_data_file = "paraloq_data.jsonl"
    
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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Paraloq handler."""
        super().__init__(config)
        
        # Override data dir and file to use existing structured task data
        self.data_dir = DATA_DIR
        self.data_file = DATA_DIR / "paraloq_data.jsonl"
        self.cache_dir = CACHE_DIR

    def load_dataset(self):
        """Load dataset from preprocessed JSONL file.
        
        Auto-downloads if not present using the existing download utilities.
        
        Returns:
            List of preprocessed samples
        """
        # Check if data file exists
        if not self.data_file.exists():
            logger.info(f"Dataset not found. Downloading to {self.data_file}")
            self._download_dataset()
        
        # Load from JSONL
        logger.info(f"Loading dataset from {self.data_file}")
        samples = []
        
        with open(self.data_file, "r") as f:
            for line in f:
                samples.append(json.loads(line))
        
        logger.info(f"Loaded {len(samples)} samples")
        return samples

    def _download_dataset(self):
        """Download and preprocess the dataset using existing utilities."""
        from ..structured.utils.dataset_download import download_and_preprocess_dataset
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        download_and_preprocess_dataset(
            dataset_name="paraloq/json_data_extraction",
            output_file=self.data_file,
            cache_dir=str(self.cache_dir),
            split="train",
            max_input_chars=20000,
        )

    def preprocess_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """Transform a raw sample to eval format.
        
        The paraloq data is already preprocessed, so we just pass it through.
        
        Args:
            raw_sample: Raw sample from JSONL (already preprocessed)
            idx: Sample index
            
        Returns:
            Sample dict ready for evaluation
        """
        # Data is already preprocessed with id, text, schema, expected
        return raw_sample

    def get_prompt(self, sample: Dict[str, Any]) -> tuple[str, str]:
        """Build prompts for structured extraction.
        
        Args:
            sample: Sample dict with text and schema
            
        Returns:
            Tuple of (system_prompt, user_prompt)
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

