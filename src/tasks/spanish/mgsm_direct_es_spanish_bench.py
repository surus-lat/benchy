"""MGSM (Multilingual Grade School Math) Spanish task."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..common import FreeformHandler
from ..common import CachedTSVMixin


class MgsmDirectEsSpanishBench(CachedTSVMixin, FreeformHandler):
    """MGSM Direct Spanish task: Math word problems."""
    
    name = "mgsm_direct_es_spanish_bench"
    display_name = "MGSM Spanish"
    description = "Multilingual Grade School Math in Spanish"
    
    dataset_name = "juletxara/mgsm"
    split = "test"
    dataset_file = "test.jsonl"
    tsv_filename = "mgsm_es.tsv"
    
    system_prompt = ""
    user_prompt_template = "{text}"
    
    def _download_tsv_and_cache(self, output_path: Path):
        """
        Download the MGSM TSV from the Hugging Face dataset and save it as a JSONL file of prompt/answer records.
        
        Each non-empty TSV line with at least two tab-separated fields is converted into a record with keys:
        - `id`: "mgsm_{index}"
        - `text`: "Pregunta: {question}\nRespuesta: "
        - `expected`: the answer as a string
        
        Lines with fewer than two fields are skipped.
        
        Parameters:
        	output_path (Path): Destination path for the output JSONL file.
        
        Raises:
        	ImportError: If the `huggingface_hub` package is not installed (message: "huggingface_hub required. Install with: pip install huggingface_hub").
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub required. Install with: pip install huggingface_hub")
        
        from ..common import save_to_jsonl
        
        tsv_path = hf_hub_download(
            repo_id=self.dataset_name,
            filename=self.tsv_filename,
            repo_type="dataset",
            cache_dir=str(self.data_dir / "cache"),
        )
        
        processed = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split("\t")
                if len(parts) >= 2:
                    question = parts[0].strip()
                    answer_number = parts[1].strip()
                    
                    prompt_text = f"Pregunta: {question}\nRespuesta: "
                    
                    processed.append({
                        "id": f"mgsm_{len(processed)}",
                        "text": prompt_text,
                        "expected": str(answer_number),
                    })
        
        save_to_jsonl(processed, output_path)