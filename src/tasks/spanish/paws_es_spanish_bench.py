"""PAWS-es (Paraphrase Adversaries from Word Scrambling) Spanish task."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common import MultipleChoiceHandler
from ..common import CachedDatasetMixin


def _general_detokenize(text: str) -> str:
    """Detokenize text by removing extra whitespace."""
    if not text:
        return text
    text = re.sub(r' +', ' ', text)
    return text.strip()


def _lowercase_first_letter(text: str) -> str:
    """Lowercase the first letter of text."""
    if not text:
        return text
    return text[0].lower() + text[1:]


def _process_paraphrase_doc(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process paraphrase document for PAWS task."""
    if doc.get("sentence1") in [None, ""] or doc.get("sentence2") in [None, ""]:
        return None
    
    sentence1 = _general_detokenize(doc["sentence1"]).strip()
    sentence2 = _general_detokenize(doc["sentence2"]).strip()
    
    # Remove final punctuation mark in the first sentence
    if sentence1.endswith((".", ",", ";")):
        sentence1 = sentence1[:-1]
    
    # Start the second sentence in lowercase
    sentence2 = _lowercase_first_letter(sentence2)
    
    return {
        "sentence1": sentence1,
        "sentence2": sentence2,
        "label": doc.get("label", 0),
        "id": doc.get("id", ""),
    }


class PawsEsSpanishBench(CachedDatasetMixin, MultipleChoiceHandler):
    """PAWS-es task: Paraphrase identification in Spanish."""
    
    name = "paws_es_spanish_bench"
    display_name = "PAWS-es"
    description = "Spanish PAWS (Paraphrase Adversaries from Word Scrambling)"
    
    dataset_name = "paws-x"
    split = "test"
    dataset_file = "test.jsonl"
    
    strict_parsing = False
    labels = {0: "No", 1: "Sí"}
    
    user_prompt_template = "{text}\n\n{choices}\n\nRespuesta:"
    
    def _download_and_cache(self, output_path: Path):
        """Transform PAWS-es dataset to eval format."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        from ..common import save_to_jsonl
        
        dataset = load_dataset(
            "paws-x", 
            "es", 
            split=self.split, 
            cache_dir=str(self.data_dir / "cache"),
        )
        
        processed = []
        for idx, raw_sample in enumerate(dataset):
            doc = _process_paraphrase_doc(raw_sample)
            if doc is None:
                continue
            
            sentence1 = doc["sentence1"]
            sentence2 = doc["sentence2"]
            label = doc["label"]
            
            text = f"{sentence1}\n{sentence2}\nPregunta: ¿Son estas frases paráfrasis?"
            
            processed.append({
                "id": doc.get("id", f"paws_es_{idx}"),
                "text": text,
                "choices": ["No", "Sí"],
                "expected": label,
            })
        
        save_to_jsonl(processed, output_path)
