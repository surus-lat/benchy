"""COPA-es (Choice of Plausible Alternatives) Spanish task."""

from pathlib import Path
from typing import Any, Dict, List

from ..common import download_huggingface_dataset, save_to_jsonl
from ..common import MultipleChoiceHandler
from ..common import CachedDatasetMixin


def _lowercase_first_letter(text: str) -> str:
    """
    Return the input string with its first character converted to lowercase.
    
    Returns:
        str: The original string with its first character lowercased; empty input is returned unchanged.
    """
    if not text:
        return text
    return text[0].lower() + text[1:]


class CopaEs(CachedDatasetMixin, MultipleChoiceHandler):
    """COPA-es task: Causal reasoning in Spanish."""
    
    name = "copa_es"
    display_name = "COPA-es"
    description = "Spanish COPA (Choice of Plausible Alternatives)"
    
    dataset_name = "BSC-LT/COPA-es"
    split = "test"
    dataset_file = "test.jsonl"
    
    # Use permissive parsing for Spanish
    strict_parsing = False
    
    # Dummy labels (choices are dynamic)
    labels = {0: "Choice 1", 1: "Choice 2"}
    user_prompt_template = "{text}\n\nOpciones:\n{choices}\n\nRespuesta:"
    
    def _download_and_cache(self, output_path: Path):
        """
        Download the COPA-es dataset, transform each example into evaluation format, and save as JSONL.
        
        Each processed example contains an `id`, a `text` composed from the premise and a Spanish connective derived from the question, a `choices` list with each choice's first letter lowercased, and an `expected` label. The resulting list is written to `output_path` as JSONL.
        
        Parameters:
            output_path (Path): Destination file path where the transformed JSONL will be saved.
        """
        raw_samples = download_huggingface_dataset(
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=str(self.data_dir / "cache"),
        )
        
        processed = []
        for idx, raw_sample in enumerate(raw_samples):
            # Process choices: lowercase first letter
            choice1 = _lowercase_first_letter(raw_sample.get("choice1", ""))
            choice2 = _lowercase_first_letter(raw_sample.get("choice2", ""))
            
            # Build text from premise + question
            premise = raw_sample.get("premise", "").rstrip(".!?,").strip()
            question = raw_sample.get("question", "")
            question_word = "porque" if question == "cause" else "y por lo tanto"
            text = f"{premise} {question_word}"
            
            processed.append({
                "id": raw_sample.get("idx", f"copa_es_{idx}"),
                "text": text,
                "choices": [choice1, choice2],
                "expected": raw_sample.get("label", 0),
            })
        
        save_to_jsonl(processed, output_path)