"""Environmental claims classification subtask.

Binary classification task for detecting environmental claims in text.
"""

from ..common import MultipleChoiceHandler


class EnvironmentalClaims(MultipleChoiceHandler):
    """Environmental claim detection (binary classification).
    
    This task evaluates models on their ability to detect whether a given
    sentence contains an environmental claim or not.
    """

    # Task metadata
    name = "environmental_claims"
    display_name = "Environmental Claims"
    description = "Binary classification for environmental claim detection"

    # Dataset configuration
    dataset = "climatebert/environmental_claims"
    split = "test"
    text_field = "text"
    label_field = "label"
    labels = {0: "No", 1: "Yes"}

    # Prompts
    system_prompt = "You are a helpful assistant for environmental claim detection."

    def get_prompt(self, sample):
        """
        Builds the system and user prompts for the environmental claim classification task.
        
        Parameters:
            sample (dict): Input example containing at least the "text" field and optionally
                "choices" (list) and "choice_labels" (mapping) used to render the label options.
        
        Returns:
            tuple: (system_prompt, user_prompt) where `system_prompt` is the task system message
            and `user_prompt` is the formatted question including the sentence and label choices.
        """
        from ..common import format_choices

        choices_text = format_choices(
            sample.get("choices", []), sample.get("choice_labels")
        )

        user_prompt = (
            f"Is there an environmental claim in the sentence?\n\n"
            f"Sentence:\n{sample.get('text', '')}\n\n"
            f"Labels:\n{choices_text}\n\n"
            f"Answer (label only):"
        )

        return self.system_prompt, user_prompt
