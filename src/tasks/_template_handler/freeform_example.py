"""Freeform text generation example subtask using handler system.

This example shows how to create a freeform text generation task.
Replace this with your actual task implementation.
"""

from ..common import FreeformHandler, ExactMatch, F1Score


class FreeformExample(FreeformHandler):
    """Example freeform text generation task.
    
    This task demonstrates the minimal configuration needed for a
    freeform text generation task using the handler system.
    
    To use this template:
    1. Change the class name to match your task (in PascalCase)
    2. Update the dataset path and configuration
    3. Choose appropriate metrics
    4. Customize prompts
    """

    # Dataset configuration
    # Replace with your HuggingFace dataset path
    dataset = "org/your-generation-dataset"
    split = "test"
    text_field = "input"  # field name for input text
    label_field = "output"  # field name for expected output

    # Prompts
    system_prompt = "You are a helpful assistant for text generation."
    user_prompt_template = "{input}\n\nGenerate:"

    # Metrics - choose what's appropriate for your task
    metrics = [
        ExactMatch(),
        F1Score(),
        # Add more metrics as needed:
        # BLEUScore(),
        # ROUGEScore(),
    ]

    # Text normalization options
    normalize_prediction = True  # Normalize whitespace, etc.
    case_sensitive = False  # Case-insensitive comparison

    # Optional: Custom prompt generation
    # Uncomment only if the default prompt template isn't sufficient
    # def get_prompt(self, sample):
    #     """Build prompt for text generation.
    #     
    #     Args:
    #         sample: Sample dict with input text
    #         
    #     Returns:
    #         Tuple of (system_prompt, user_prompt)
    #     """
    #     user_prompt = f"Input: {sample.get('input', '')}\n\nGenerate appropriate output:"
    #     return self.system_prompt, user_prompt

    # Optional: Custom preprocessing
    # Uncomment only if you need to transform the dataset format
    # def preprocess_sample(self, raw_sample, idx):
    #     """Transform raw sample to eval format.
    #     
    #     Args:
    #         raw_sample: Raw sample from dataset
    #         idx: Sample index
    #         
    #     Returns:
    #         Processed sample or None to skip
    #     """
    #     # Extract fields
    #     input_text = raw_sample.get(self.text_field)
    #     output_text = raw_sample.get(self.label_field)
    #
    #     if not input_text or not output_text:
    #         return None  # Skip invalid samples
    #
    #     return {
    #         "id": f"{self.get_task_name()}_{idx}",
    #         "input": str(input_text),
    #         "expected": str(output_text),
    #     }

