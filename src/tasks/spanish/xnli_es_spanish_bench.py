"""XNLI-es (Cross-lingual Natural Language Inference) Spanish task."""

from pathlib import Path
from typing import Any, Dict, List

from ..common import save_to_jsonl
from ..common import MultipleChoiceHandler
from ..common import CachedDatasetMixin


class XnliEsSpanishBench(CachedDatasetMixin, MultipleChoiceHandler):
    """XNLI-es task: Cross-lingual NLI in Spanish."""
    
    name = "xnli_es_spanish_bench"
    display_name = "XNLI Spanish"
    description = "Cross-lingual Natural Language Inference in Spanish"
    
    dataset_name = "xnli"
    split = "test"
    dataset_file = "test.jsonl"
    
    strict_parsing = False
    labels = {0: "implicación", 1: "neutral", 2: "contradicción"}
    
    user_prompt_template = "{text}\n\nOpciones:\n{choices}\n\nRespuesta:"
    
    def _download_and_cache(self, output_path: Path):
        """
        Convert the XNLI Spanish test split into a multiple-choice evaluation format and save it as a JSONL file.
        
        Loads the XNLI Spanish test dataset, transforms each sample into a dict with keys `id`, `text`, `choices`, and `expected`, and writes the resulting list to `output_path` in JSONL format.
        
        Parameters:
            output_path (Path): Destination file path for the generated JSONL.
        
        Raises:
            ImportError: If the `datasets` library is not installed.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "xnli", 
            "es", 
            split="test", 
            cache_dir=str(self.data_dir / "cache"),
            trust_remote_code=True
        )
        
        processed = []
        for idx, raw_sample in enumerate(dataset):
            premise = raw_sample.get("premise", "")
            hypothesis = raw_sample.get("hypothesis", "")
            label = raw_sample.get("label", 0)
            
            text = f"{premise}\nPregunta: {hypothesis} ¿Verdadero, Falso o Ni idea?\nRespuesta:"
            
            processed.append({
                "id": f"xnli_es_{idx}",
                "text": text,
                "choices": ["implicación", "neutral", "contradicción"],
                "expected": label,
            })
        
        save_to_jsonl(processed, output_path)
