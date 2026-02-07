"""WNLI-es (Winograd Natural Language Inference) Spanish task."""

import csv
from pathlib import Path

from ..common import MultipleChoiceHandler
from ..common import CachedCSVMixin


class WnliEs(CachedCSVMixin, MultipleChoiceHandler):
    """WNLI-es task: Winograd NLI in Spanish."""
    
    name = "wnli_es"
    display_name = "WNLI-es"
    description = "Spanish WNLI (Winograd Natural Language Inference)"
    
    dataset_name = "PlanTL-GOB-ES/wnli-es"
    split = "validation"
    dataset_file = "validation.jsonl"
    csv_filename = "wnli-dev-es.csv"
    
    strict_parsing = False
    labels = {0: "Falso", 1: "Verdadero"}
    
    user_prompt_template = "{text}\n\nOpciones:\n{choices}\n\nRespuesta:"
    
    def _download_csv_and_cache(self, output_path: Path):
        """Download and transform WNLI-es CSV to JSONL."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub required. Install with: pip install huggingface_hub")
        
        from ..common import save_to_jsonl
        
        # Map split names to CSV filenames
        split_to_file = {
            "validation": "wnli-dev-es.csv",
            "dev": "wnli-dev-es.csv",
            "train": "wnli-train-es.csv",
            "test": "wnli-test-shuffled-es.csv",
        }
        
        csv_filename = split_to_file.get(self.split, self.csv_filename)
        
        csv_path = hf_hub_download(
            repo_id=self.dataset_name,
            filename=csv_filename,
            repo_type="dataset",
            cache_dir=str(self.data_dir / "cache"),
        )
        
        processed = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sentence1 = row.get("sentence1", "").strip()
                sentence2 = row.get("sentence2", "").strip()
                try:
                    label = int(row.get("label", 0))
                except ValueError:
                    continue  # Skip malformed rows
                index = row.get("index", "")
                
                if not sentence1 or not sentence2:
                    continue
                
                text = f"{sentence1}\nPregunta: {sentence2} ¿Verdadero o Falso?\nRespuesta:"
                
                processed.append({
                    "id": f"wnli_es_{index}" if index else f"wnli_es_{len(processed)}",
                    "text": text,
                    "choices": ["Falso", "Verdadero"],
                    "expected": label,
                })
        
        save_to_jsonl(processed, output_path)
