"""Email extraction subtask - customer support email parsing."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..common import StructuredHandler, CachedDatasetMixin, download_huggingface_dataset, save_to_jsonl

logger = logging.getLogger(__name__)


class EmailExtract(CachedDatasetMixin, StructuredHandler):
    """Extract structured data from customer support emails.
    
    This task extracts customer information, order details, and issues
    from Spanish-language customer support emails.
    """

    # Task configuration
    name = "email_extract"
    display_name = "Email Data Extraction"
    description = "Extract structured customer data from support emails"
    
    # Dataset configuration
    dataset_name = "mauroibz/email_extract"
    split = "train"
    dataset_file = "email_extract_data.jsonl"
    
    # Field mapping
    text_field = "email_content"
    label_field = "extracted_data"
    
    # Schema for customer support email extraction
    schema = {
        "type": "object",
        "description": "Structured data extracted from Spanish customer support emails",
        "properties": {
            "customer_name": {
                "type": "string"
            },
            "email": {
                "type": "string"
            },
            "order_id": {
                "type": "string"
            },
            "product": {
                "type": "string"
            },
            "quantity": {
                "type": "integer"
            },
            "issue": {
                "type": "string",
                "description": "Detailed description of the problem or issue reported by the customer"
            }
        },
        "required": ["customer_name", "email", "order_id", "product", "quantity", "issue"],
        "additionalProperties": False
    }
    
    # Prompts
    system_prompt = (
        "You are a precise data extraction assistant for customer support. "
        "Extract information from customer emails according to the given JSON schema. "
        "Only extract information explicitly stated in the email."
    )
    
    def _download_and_cache(self, output_path: Path):
        """
        Download the source dataset, normalize each sample to the task schema, and write the processed samples to a JSON Lines file.
        
        Parameters:
            output_path (Path): File path where the processed samples will be saved in JSON Lines format. Each record contains the keys `id` (six-digit zero-padded identifier), `text` (email content), `schema` (the JSON schema used for extraction), and `expected` (the labeled extraction from the source sample).
        """
        raw_samples = download_huggingface_dataset(
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=str(self.data_dir / "cache"),
        )
        
        processed = []
        for idx, raw_sample in enumerate(raw_samples):
            text = raw_sample.get(self.text_field, "")
            expected = raw_sample.get(self.label_field, {})
            
            processed.append({
                "id": f"{idx+1:06d}",
                "text": text,
                "schema": self.schema,
                "expected": expected,
            })
        
        save_to_jsonl(processed, output_path)
        logger.info(f"Cached {len(processed)} samples to {output_path}")
    
    def preprocess_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """
        Return the input sample unchanged because preprocessing is performed in _download_and_cache.
        
        Parameters:
            raw_sample (Dict[str, Any]): The raw dataset sample.
            idx (int): Index of the sample in the dataset; ignored by this implementation.
        
        Returns:
            Dict[str, Any]: The original `raw_sample` unchanged.
        """
        return raw_sample