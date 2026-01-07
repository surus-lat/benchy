"""Multiple choice example subtask using handler system.

This example shows how to create a multiple choice task with minimal code.
Replace this with your actual task implementation.
"""

from ..formats import MultipleChoiceHandler


class McqExample(MultipleChoiceHandler):
    """Example multiple choice task.
    
    This task demonstrates the minimal configuration needed for a
    multiple choice classification task using the handler system.
    
    To use this template:
    1. Change the class name to match your task (in PascalCase)
    2. Update the dataset path and configuration
    3. Set appropriate labels
    4. Customize prompts
    5. Optionally override get_prompt() for dynamic prompts
    """

    # Dataset configuration
    # Replace with your HuggingFace dataset path
    dataset = "org/your-mcq-dataset"
    split = "test"  # or "train", "validation", etc.
    text_field = "text"  # field name for input text
    label_field = "label"  # field name for the label
    
    # Label mapping: {value: display_text}
    labels = {
        0: "Option A",
        1: "Option B",
        2: "Option C",
    }

    # Prompts
    system_prompt = "You are a helpful assistant for multiple choice questions."
    
    # Optional: Override user_prompt_template
    # user_prompt_template = "{text}\n\nChoose the correct option."

    # Optional: Customize prompt generation
    def get_prompt(self, sample):
        """Build prompt for this subtask.
        
        Args:
            sample: Sample dict with text, choices, choice_labels
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        from ...common import format_choices

        # Format choices nicely
        choices_text = format_choices(
            sample.get("choices", []),
            sample.get("choice_labels")
        )

        # Build user prompt
        user_prompt = (
            f"Question: {sample.get('text', '')}\n\n"
            f"Options:\n{choices_text}\n\n"
            f"Answer (label only):"
        )

        return self.system_prompt, user_prompt

