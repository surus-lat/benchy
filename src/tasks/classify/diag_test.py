"""Clinical diagnostic classification subtask.

Multi-class classification task for identifying personality disorder diagnoses
based on symptoms.
"""

from ..common import MultipleChoiceHandler


class DiagTest(MultipleChoiceHandler):
    """Clinical diagnostic category classification.
    
    This task evaluates models on their ability to classify personality disorder
    categories based on symptom descriptions.
    """

    # Task metadata
    name = "diag_test"
    display_name = "DiagTest"
    description = "Clinical diagnostic category classification"

    # Dataset configuration
    dataset = "somosnlp-hackathon-2023/DiagTrast"
    split = "train"
    text_field = "Sintoma"
    label_field = "Padecimiento_cat"
    labels = {
        0: "Trastornos de la personalidad antisocial",
        1: "Trastornos de la personalidad borderline",
        2: "Trastornos de la personalidad esquizotipica",
        3: "Trastornos de la personalidad histrionica",
        4: "Trastornos de la personalidad narcisista",
    }

    # Prompts
    system_prompt = "You are a helpful assistant for clinical category labeling."

    def get_prompt(self, sample):
        """Build prompt for diagnostic classification.
        
        Args:
            sample: Sample dict with text and choices
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        from ..common import format_choices

        choices_text = format_choices(
            sample.get("choices", []), sample.get("choice_labels")
        )

        user_prompt = (
            f"Elije el diagnostico mas probable para el sintoma:\n\n"
            f"Sintoma:\n{sample.get('text', '')}\n\n"
            f"Labels:\n{choices_text}\n\n"
            f"Answer (label only):"
        )

        return self.system_prompt, user_prompt

