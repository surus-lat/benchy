"""Teleia PCE (Prueba de Conocimientos Específicos) Spanish task."""

import re
from pathlib import Path
from typing import Any, Dict, List

from ..common import MultipleChoiceHandler
from ..common import CachedDatasetMixin


def _preprocess_text(text: str) -> str:
    """
    Normalize Teleia PCE text by removing editorial markers and collapsing extra whitespace.
    
    Returns:
        str: The cleaned text with markers removed and redundant spaces collapsed.
    """
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _process_pce_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw Teleia PCE document into a standardized multiple-choice example.
    
    Parameters:
        doc (Dict[str, Any]): Raw Teleia PCE document, expected to contain keys
            "question", "option_a", "option_b", "option_c", and "answer".
    
    Returns:
        Dict[str, Any]: Dictionary with keys:
            - "query" (str): Prompt formed from the preprocessed question (e.g., "Pregunta: {question}\nRespuesta:").
            - "choices" (List[str]): List of preprocessed, non-empty option strings in A/B/C order.
            - "target" (int): Index of the correct choice (0 for A, 1 for B, 2 for C); defaults to 0 if the answer is missing or invalid.
    """
    question = _preprocess_text(doc.get("question", ""))
    query = f"Pregunta: {question}\nRespuesta:"
    
    choices = [
        _preprocess_text(option)
        for option in [doc.get("option_a"), doc.get("option_b"), doc.get("option_c")]
        if option
    ]
    
    answer = doc.get("answer", "A")
    target = ["A", "B", "C"].index(answer) if answer in ["A", "B", "C"] else 0
    
    return {"query": query, "choices": choices, "target": target}


class TeleiaPce(CachedDatasetMixin, MultipleChoiceHandler):
    """Teleia PCE task."""
    
    name = "teleia_pce"
    display_name = "Teleia PCE"
    description = "Teleia PCE (Prueba de Conocimientos Específicos)"
    
    dataset_name = "migonsa/teleia"
    split = "test"
    dataset_file = "test.jsonl"
    
    strict_parsing = False
    labels = {0: "A", 1: "B", 2: "C"}
    
    user_prompt_template = "{text}\n\nOpciones:\n{choices}\n\nRespuesta:"
    
    def _download_and_cache(self, output_path: Path):
        """
        Convert and cache the Teleia PCE dataset into the evaluation JSONL format.
        
        Loads the "migonsa/teleia" dataset (config "pce") using the Hugging Face datasets library, transforms each sample into a dictionary with keys `id`, `text`, `choices`, and `expected`, and writes the resulting list to `output_path` in JSONL format. Raises an ImportError with an installation hint if the `datasets` library is not available.
        
        Parameters:
            output_path (Path): Filesystem path where the processed JSONL file will be written.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        from ..common import save_to_jsonl
        
        dataset = load_dataset(
            "migonsa/teleia", 
            "pce", 
            split=self.split, 
            cache_dir=str(self.data_dir / "cache"),
            trust_remote_code=True
        )
        
        processed = []
        for idx, raw_sample in enumerate(dataset):
            doc = _process_pce_doc(raw_sample)
            
            processed.append({
                "id": raw_sample.get("id", f"teleia_pce_{idx}"),
                "text": doc["query"],
                "choices": doc["choices"],
                "expected": doc["target"],
            })
        
        save_to_jsonl(processed, output_path)