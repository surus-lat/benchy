"""PAWS-es (Paraphrase Adversaries from Word Scrambling) Spanish task."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common import MultipleChoiceHandler
from ..common import CachedDatasetMixin


def _general_detokenize(text: str) -> str:
    """
    Normalize spacing in text by collapsing consecutive spaces and trimming surrounding whitespace.
    
    Returns the input unchanged if it is falsy.
    
    Parameters:
        text (str): Input string to normalize.
    
    Returns:
        str: The input string with consecutive spaces replaced by a single space and leading/trailing whitespace removed.
    """
    if not text:
        return text
    text = re.sub(r' +', ' ', text)
    return text.strip()


def _lowercase_first_letter(text: str) -> str:
    """
    Lowercases the first character of the given string.
    
    Returns:
        The input string with its first character lowercased; if `text` is falsy or empty, returns it unchanged.
    """
    if not text:
        return text
    return text[0].lower() + text[1:]


def _process_paraphrase_doc(doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validate and normalize a PAWS paraphrase sample into standardized fields.
    
    Processes the input mapping `doc` by ensuring `sentence1` and `sentence2` are present and non-empty; returns `None` if either is missing or empty. When valid, detokenizes and trims both sentences, removes a trailing period, comma, or semicolon from `sentence1`, and lowercases the first character of `sentence2`. Also preserves `label` (defaulting to 0 if absent) and `id` (defaulting to an empty string if absent).
    
    Parameters:
        doc (Dict[str, Any]): Raw sample expected to contain `sentence1` and `sentence2`; may include optional `label` and `id`.
    
    Returns:
        Optional[Dict[str, Any]]: A dictionary with keys:
            - `sentence1` (str): Processed first sentence without trailing ., or ;.
            - `sentence2` (str): Processed second sentence with its first character lowercased.
            - `label` (int): Original label or 0 if missing.
            - `id` (str): Original id or empty string if missing.
    """
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
        """
        Download the PAWS-es split and write a transformed evaluation JSONL to output_path.
        
        Loads the Spanish PAWS-X test split, normalizes and filters examples with _process_paraphrase_doc, and writes a list of records with keys `id`, `text` (two sentences plus the Spanish paraphrase question), `choices` (`["No", "Sí"]`), and `expected` (label) to the given output path in JSONL format.
        
        Parameters:
            output_path (Path): Filesystem path where the resulting JSONL will be written.
        
        Raises:
            ImportError: If the `datasets` library is not available.
        """
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
            trust_remote_code=True
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