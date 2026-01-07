"""Structured extraction example subtask using handler system.

This example shows how to create a structured extraction task.
Replace this with your actual task implementation.
"""

from ..formats import StructuredHandler


class StructuredExample(StructuredHandler):
    """Example structured extraction task.
    
    This task demonstrates the minimal configuration needed for a
    structured extraction task using the handler system.
    
    To use this template:
    1. Change the class name to match your task (in PascalCase)
    2. Update the dataset path and configuration
    3. Customize prompts
    4. Optionally configure metrics
    """

    # Dataset configuration
    # Replace with your HuggingFace dataset path
    dataset = "org/your-extraction-dataset"
    split = "test"
    text_field = "text"  # field name for input text
    schema_field = "schema"  # field name for JSON schema
    label_field = "expected"  # field name for expected output

    # Prompts
    system_prompt = "You are an expert at extracting structured information."
    
    # Optional: Customize metrics configuration
    metrics_config = {
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
                "absolute_tolerance": 1e-6,
            },
        },
        "extraction_quality_score": {
            "enabled": True,
            "weights": {
                "schema_validity": 0.2,
                "field_f1_partial": 0.6,
                "inverted_hallucination": 0.2,
            },
        },
    }

    # Optional: Custom prompt generation
    def get_prompt(self, sample):
        """Build prompt for structured extraction.
        
        Args:
            sample: Sample dict with text and schema
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        import json

        schema = sample.get("schema", {})
        schema_str = json.dumps(schema, indent=2) if schema else ""

        user_prompt = (
            f"Extract information from the following text:\n\n"
            f"{sample.get('text', '')}\n\n"
            f"Follow this JSON schema:\n{schema_str}\n\n"
            f"Return valid JSON matching the schema exactly."
        )

        return self.system_prompt, user_prompt

