"""Teleia PCE (Prueba de Conocimientos Específicos) Spanish task."""

import re
from pathlib import Path
from typing import Any, Dict, List

from ..common import MultipleChoiceHandler
from ..common import CachedDatasetMixin


def _preprocess_text(text: str) -> str:
    """Preprocess teleia text by removing special markers."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _process_pce_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Process PCE document."""
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
        """Transform Teleia PCE dataset to eval format."""
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("datasets library required. Install with: pip install datasets") from e
        
        from ..common import save_to_jsonl
        
        dataset = load_dataset(
            "migonsa/teleia", 
            "pce", 
            split=self.split, 
            cache_dir=str(self.data_dir / "cache"),
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
