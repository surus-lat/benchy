"""Teleia Cervantes AVE Spanish task."""

import re
from pathlib import Path
from typing import Any, Dict, List

from ..common import MultipleChoiceHandler
from ..common import CachedDatasetMixin


def _preprocess_text(text: str) -> str:
    """
    Clean and normalize a Cervantes AVE text string.
    
    Performs the following transformations: trims leading/trailing whitespace, replaces the marker " [title]" with ". ", removes any content enclosed in square brackets, and collapses consecutive double spaces into single spaces.
    
    Returns:
        The cleaned text string.
    """
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _process_cervantes_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw Cervantes AVE record into a standardized multiple-choice item.
    
    Parameters:
        doc (Dict[str, Any]): Raw dataset record containing at least question and option fields
            (e.g., "option_a" through "option_d") and an "answer" letter.
    
    Returns:
        Dict[str, Any]: A mapping with:
            - "query" (str): Prompt constructed from the preprocessed question.
            - "choices" (List[str]): List of preprocessed, non-empty answer options.
            - "target" (int): Index (0–3) of the correct choice corresponding to A–D; defaults to 0 if unspecified or invalid.
    """
    question = _preprocess_text(doc.get("question", ""))
    query = f"Pregunta: {question}\nRespuesta:"
    
    choices = [
        _preprocess_text(option)
        for option in [
            doc.get("option_a"),
            doc.get("option_b"),
            doc.get("option_c"),
            doc.get("option_d"),
        ]
        if option
    ]
    
    answer = doc.get("answer", "A")
    target = ["A", "B", "C", "D"].index(answer) if answer in ["A", "B", "C", "D"] else 0
    
    return {"query": query, "choices": choices, "target": target}


class TeleiaCervantesAve(CachedDatasetMixin, MultipleChoiceHandler):
    """Teleia Cervantes AVE task."""
    
    name = "teleia_cervantes_ave"
    display_name = "Teleia Cervantes AVE"
    description = "Teleia Cervantes AVE assessment"
    
    dataset_name = "migonsa/teleia"
    split = "test"
    dataset_file = "test.jsonl"
    
    strict_parsing = False
    labels = {0: "A", 1: "B", 2: "C", 3: "D"}
    
    user_prompt_template = "{text}\n\nOpciones:\n{choices}\n\nRespuesta:"
    
    def _download_and_cache(self, output_path: Path):
        """
        Download the Cervantes AVE subset of the Teleia dataset, convert it to evaluation format, and write it to the given path.
        
        This fetches the remote dataset, transforms each sample into a standardized dict with keys "id", "text", "choices", and "expected", and saves the collection as a JSONL file at output_path.
        
        Parameters:
            output_path (Path): Filesystem path where the resulting JSONL file will be written.
        
        Raises:
            ImportError: If the `datasets` library is not installed.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        from ..common import save_to_jsonl
        
        dataset = load_dataset(
            "migonsa/teleia", 
            "cervantes_ave", 
            split=self.split, 
            cache_dir=str(self.data_dir / "cache"),
            trust_remote_code=True
        )
        
        processed = []
        for idx, raw_sample in enumerate(dataset):
            doc = _process_cervantes_doc(raw_sample)
            
            processed.append({
                "id": raw_sample.get("id", f"teleia_cervantes_ave_{idx}"),
                "text": doc["query"],
                "choices": doc["choices"],
                "expected": doc["target"],
            })
        
        save_to_jsonl(processed, output_path)