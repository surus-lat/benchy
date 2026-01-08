"""OpenBookQA-es (Spanish OpenBookQA) task."""

from pathlib import Path
from typing import Any, Dict, List

from ..common import download_huggingface_dataset, save_to_jsonl
from ..common import MultipleChoiceHandler
from ..common import CachedDatasetMixin


class OpenbookqaEs(CachedDatasetMixin, MultipleChoiceHandler):
    """OpenBookQA-es task: Open-domain QA in Spanish."""
    
    name = "openbookqa_es"
    display_name = "OpenBookQA-es"
    description = "Spanish OpenBookQA"
    
    dataset_name = "BSC-LT/openbookqa-es"
    split = "test"
    dataset_file = "test.jsonl"
    
    strict_parsing = False
    labels = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    
    user_prompt_template = "{text}\n\nOpciones:\n{choices}\n\nRespuesta:"
    
    def _download_and_cache(self, output_path: Path):
        """Transform OpenBookQA-es dataset to eval format."""
        raw_samples = download_huggingface_dataset(
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=str(self.data_dir / "cache"),
        )
        
        processed = []
        for idx, raw_sample in enumerate(raw_samples):
            question_stem = raw_sample.get("question_stem", "")
            answer_key = raw_sample.get("answerKey", "").strip()
            
            choices_data = raw_sample.get("choices", {})
            choices_text = choices_data.get("text", [])
            choices_labels = choices_data.get("label", [])
            
            try:
                expected_idx = choices_labels.index(answer_key)
            except (ValueError, AttributeError):
                expected_idx = 0
            
            processed.append({
                "id": raw_sample.get("id", f"openbookqa_es_{idx}"),
                "text": question_stem,
                "choices": choices_text,
                "expected": expected_idx,
            })
        
        save_to_jsonl(processed, output_path)
