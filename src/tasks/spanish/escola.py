"""EsCoLA (Spanish Corpus of Linguistic Acceptability) task."""

from pathlib import Path

from ..common import download_huggingface_dataset, save_to_jsonl
from ..common import MultipleChoiceHandler
from ..common import CachedDatasetMixin


class Escola(CachedDatasetMixin, MultipleChoiceHandler):
    """EsCoLA task: Spanish grammatical acceptability."""
    
    name = "escola"
    display_name = "EsCoLA"
    description = "Spanish Corpus of Linguistic Acceptability"
    
    dataset_name = "nbel/EsCoLA"
    split = "validation"
    dataset_file = "validation.jsonl"
    
    strict_parsing = False
    labels = {0: "No", 1: "Sí"}
    
    user_prompt_template = "{text}\n\nOpciones:\n{choices}\n\nRespuesta:"
    
    def _download_and_cache(self, output_path: Path):
        """Transform EsCoLA dataset to eval format."""
        raw_samples = download_huggingface_dataset(
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=str(self.data_dir / "cache"),
        )
        
        processed = []
        for idx, raw_sample in enumerate(raw_samples):
            sentence = raw_sample.get("Sentence", "")
            label = raw_sample.get("Label", 0)
            
            text = f"{sentence}\nPregunta: ¿Tiene sentido esta frase?\nRespuesta:"
            
            processed.append({
                "id": raw_sample.get("idx", f"escola_{idx}"),
                "text": text,
                "choices": ["No", "Sí"],
                "expected": label,
            })
        
        save_to_jsonl(processed, output_path)
