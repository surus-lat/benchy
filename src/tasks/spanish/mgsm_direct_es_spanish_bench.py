"""MGSM (Multilingual Grade School Math) Spanish task."""

from pathlib import Path

from ..common import FreeformHandler
from ..common import CachedDatasetMixin


class MgsmDirectEsSpanishBench(CachedDatasetMixin, FreeformHandler):
    """MGSM Direct Spanish task: Math word problems."""

    name = "mgsm_direct_es_spanish_bench"
    display_name = "MGSM Spanish"
    description = "Multilingual Grade School Math in Spanish"

    dataset_name = "juletxara/mgsm"
    split = "test"
    # NOTE: must NOT be "test.jsonl" — copa_es.py already caches COPA data there
    # (both tasks share .data/spanish/); sharing the filename made MGSM evaluate
    # COPA prompts (exact_match=0 for every model). See .notes/TOGETHER-BENCH-2026-09-02.md
    dataset_file = "mgsm_es_test.jsonl"

    system_prompt = ""
    user_prompt_template = "{text}"

    def _download_and_cache(self, output_path: Path):
        """Download MGSM es config and transform to eval format."""
        from datasets import load_dataset

        from ..common import save_to_jsonl

        dataset = load_dataset(
            self.dataset_name,
            "es",  # language config (repo has one config per language)
            split=self.split,
            cache_dir=str(self.data_dir / "cache"),
        )

        processed = []
        for raw_sample in list(dataset):
            question = str(raw_sample.get("question", "")).strip()
            answer_number = str(raw_sample.get("answer_number", "")).strip()
            if not question:
                continue
            # exact_match scoring: enforce the bare-number answer format the
            # metric expects (metadata: "direct numeric answers, no CoT")
            prompt_text = (
                f"Pregunta: {question}\n"
                "Responde ÚNICAMENTE con el número final, sin unidades, "
                "sin símbolos de moneda y sin explicaciones.\n"
                "Respuesta: "
            )
            processed.append({
                "id": f"mgsm_{len(processed)}",
                "text": prompt_text,
                "expected": answer_number,
            })

        save_to_jsonl(processed, output_path)