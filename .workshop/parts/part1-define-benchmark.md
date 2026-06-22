# Part 1 — Define a custom benchmark

**Time:** ~25 min  
**Output:** A new task at `src/tasks/workshop_extraction/` with custom scoring, discoverable by `benchy tasks` and runnable by `benchy eval`  
**Skill:** [`skills/workshop-define-benchmark/SKILL.md`](../skills/workshop-define-benchmark/SKILL.md)

---

## Goal

Add a new benchy task that uses the **same dataset** as the production `structured_extraction/paraloq` task but with a **different scoring function**. Same model, same data, different weights → different score. That's the lesson: scoring is a knob.

## What you're going to change vs the production paraloq task

| Aspect | Production paraloq | Your workshop task |
|---|---|---|
| Dataset | `paraloq/json_data_extraction` (train) | Same |
| Schema | Per-row, from dataset | Same |
| EQS weights | schema_validity **0.2** / field_f1_partial **0.6** / inverted_hallucination **0.2** | schema_validity **0.5** / field_f1_partial **0.3** / inverted_hallucination **0.2** |
| String matching | exact_threshold **0.95** / partial_threshold **0.5** | exact_threshold **0.99** / partial_threshold **0.8** |

The reweighted EQS rewards strict schema compliance over fuzzy field matching. The tighter string thresholds penalize "close enough" matches.

---

## Step 1 — Copy the template into a new task directory

```bash
cp -r src/tasks/_template_handler src/tasks/workshop_extraction
cd src/tasks/workshop_extraction
rm freeform_example.py mcq_example.py README.md
mv structured_example.py extract.py
cd -
```

You now have:
```
src/tasks/workshop_extraction/
├── extract.py        # placeholder structured-extraction handler from the template
└── metadata.yaml     # placeholder task metadata from the template
```

> **Naming matters.** The file `extract.py` will define a class `Extract` (PascalCase of the filename). benchy's task registry auto-discovers `<snake_case>.py` → `<PascalCase>` classes in any `src/tasks/<group>/` directory.

---

## Step 2 — Rewrite `extract.py`

Open `src/tasks/workshop_extraction/extract.py`. Replace **everything** in the file with this:

```python
"""Workshop extraction task — paraloq dataset with custom (stricter) scoring."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..common import (
    CachedDatasetMixin,
    StructuredHandler,
    download_huggingface_dataset,
    save_to_jsonl,
)

logger = logging.getLogger(__name__)


class Extract(CachedDatasetMixin, StructuredHandler):
    """Workshop extraction subtask with reweighted EQS and stricter string matching."""

    name = "extract"
    display_name = "Workshop Extraction"
    description = "Paraloq with reweighted EQS — workshop demo"

    dataset_name = "paraloq/json_data_extraction"
    split = "train"
    dataset_file = "workshop_data.jsonl"

    system_prompt = (
        "You are a precise data extraction assistant. Extract information from "
        "the provided text according to the given JSON schema. Only extract "
        "information explicitly stated. Use null when absent."
    )

    # ▼▼▼ THIS BLOCK IS THE "CUSTOM SCORING" PART — these are the knobs ▼▼▼
    metrics_config = {
        "extraction_quality_score": {
            "enabled": True,
            "weights": {
                "schema_validity": 0.5,           # production: 0.2
                "field_f1_partial": 0.3,          # production: 0.6
                "inverted_hallucination": 0.2,    # production: 0.2
            },
        },
        "partial_matching": {
            "string": {
                "token_overlap_weight": 0.5,
                "levenshtein_weight": 0.3,
                "containment_weight": 0.2,
                "exact_threshold": 0.99,          # production: 0.95
                "partial_threshold": 0.80,        # production: 0.5
            },
            "number": {
                "relative_tolerance": 0.001,
                "absolute_tolerance": 1e-06,
            },
            "array": {"method": "jaccard", "partial_credit": True},
        },
        "normalization": {
            "case_sensitive": False,
            "normalize_whitespace": True,
            "unicode_normalize": True,
        },
    }
    # ▲▲▲ END CUSTOM SCORING ▲▲▲

    def _download_and_cache(self, output_path: Path) -> None:
        raw_samples = download_huggingface_dataset(
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=str(self.data_dir / "cache"),
        )

        processed = []
        for idx, raw in enumerate(raw_samples):
            text = raw.get("text", "")
            try:
                schema = json.loads(raw["schema"]) if isinstance(raw.get("schema"), str) else raw.get("schema", {})
                expected = json.loads(raw["item"]) if isinstance(raw.get("item"), str) else raw.get("item", {})
            except json.JSONDecodeError:
                logger.warning(f"Sample {idx}: failed to parse schema/item, skipping")
                continue

            if len(text) > 20000:
                text = text[:20000]

            processed.append({
                "id": f"{idx:06d}",
                "text": text,
                "schema": schema,
                "expected": expected,
            })

        save_to_jsonl(processed, output_path)

    def preprocess_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        return raw_sample

    def get_prompt(self, sample: Dict[str, Any]) -> tuple[str, str]:
        schema = sample.get("schema", {})
        schema_str = json.dumps(schema, indent=2) if schema else ""
        user_prompt = (
            f"Text:\n{sample.get('text', '')}\n\n"
            f"Extract information according to this JSON schema:\n"
            f"{schema_str}\n\n"
            f"Output valid JSON matching the schema exactly."
        )
        return self.system_prompt, user_prompt
```

> **Shortcut if you'd rather not type:**  
> `cp .workshop/assets/workshop_extraction_extract.py.example src/tasks/workshop_extraction/extract.py`

### What changed vs the template

| Block | Template had | You wrote |
|---|---|---|
| Class inheritance | `class StructuredExample(StructuredHandler):` | `class Extract(CachedDatasetMixin, StructuredHandler):` (the mixin is what makes HuggingFace downloads work without errors) |
| Dataset attribute | `dataset = "org/your-extraction-dataset"` | `dataset_name = "paraloq/json_data_extraction"` (the mixin uses `dataset_name`, not `dataset`) |
| `dataset_file` | not set | `dataset_file = "workshop_data.jsonl"` (required by the mixin — where to cache) |
| `_download_and_cache` | not implemented | implemented (paraloq stores `schema` and `item` as JSON-encoded strings, so we have to `json.loads` them) |
| `metrics_config` | default 0.2/0.6/0.2 weights | reweighted 0.5/0.3/0.2 — the customization |
| `get_prompt` | commented-out example | implemented (paraloq needs the schema rendered into the user prompt) |

---

## Step 3 — Rewrite `metadata.yaml`

Open `src/tasks/workshop_extraction/metadata.yaml`. Replace **everything** in the file with this:

```yaml
name: workshop_extraction
display_name: Workshop Extraction
description: Structured extraction on paraloq with custom (stricter) scoring — workshop demo

capability_requirements:
  requires_logprobs: optional
  requires_multimodal: none
  requires_schema: preferred
  requires_files: none

subtasks:
  extract:
    description: Extract structured JSON from diverse text — custom-weighted EQS
    url: https://huggingface.co/datasets/paraloq/json_data_extraction
```

> **Shortcut:**  
> `cp .workshop/assets/workshop_extraction_metadata.yaml.example src/tasks/workshop_extraction/metadata.yaml`

### What changed vs the template

| Field | Template had | You wrote |
|---|---|---|
| `name` | `template_handler` | `workshop_extraction` (must match the directory name) |
| `subtasks` | three placeholders (`mcq_example`, `structured_example`, `freeform_example`) | one: `extract` (must match the `name` attribute on your `Extract` class) |

---

## Step 4 — Confirm auto-discovery

```bash
.venv/bin/benchy tasks | grep workshop_extraction
```

Expected output:
```
  - workshop_extraction
```

If it doesn't appear: check that `extract.py` is in `src/tasks/workshop_extraction/`, the class is named `Extract` (PascalCase of the filename), and `metadata.yaml` has `name: workshop_extraction`.

---

## Step 5 — Smoke run on a workshop Together model

(Part 2's templates will live in `configs/models/`. If you haven't done Part 2's setup step yet, do it now: `cp .workshop/assets/together_modelA.yaml.template configs/models/together_modelA.yaml`.)

```bash
.venv/bin/benchy eval \
  --config configs/models/together_modelA.yaml \
  --tasks workshop_extraction \
  --limit 5 --exit-policy smoke --run-id w1_smoke
```

Wall-clock: **~30-60s** on slot A.

---

## Step 6 — Read your score

```bash
cat outputs/benchmark_outputs/w1_smoke_LIMITED/*/run_outcome.json \
  | jq '{status, task: .tasks.workshop_extraction.status, summary: .tasks.workshop_extraction.summary}'

cat outputs/benchmark_outputs/w1_smoke_LIMITED/*/run_summary.json \
  | jq '.tasks.workshop_extraction.extraction_quality_score'
```

Expected: `status: "passed"`, EQS somewhere in the range **0.80 — 0.99** depending on which model you ran.

> Rehearsal got `0.984` on `google/gemma-3n-E4B-it` with 5 samples.

---

## Step 7 (optional comparison) — Run production paraloq for the same model

```bash
.venv/bin/benchy eval \
  --config configs/models/together_modelA.yaml \
  --tasks structured_extraction \
  --limit 5 --exit-policy smoke --run-id w1_smoke_prod
```

Then compare `extraction_quality_score` in the two `run_summary.json` files:

```bash
echo "Workshop (custom weights):"
cat outputs/benchmark_outputs/w1_smoke_LIMITED/*/run_summary.json \
  | jq '.tasks.workshop_extraction.extraction_quality_score'

echo "Production (default paraloq weights):"
cat outputs/benchmark_outputs/w1_smoke_prod_LIMITED/*/run_summary.json \
  | jq '.tasks.structured_extraction.paraloq.extraction_quality_score'
```

Same model, same data. Different scores. That's the lesson.

---

## What "done" looks like

- `src/tasks/workshop_extraction/extract.py` and `metadata.yaml` exist
- `benchy tasks` lists `workshop_extraction`
- `outputs/benchmark_outputs/w1_smoke_LIMITED/<model>/run_outcome.json` has `"status": "passed"`
- You can articulate why your score is different from production paraloq's

## Common stumbles

- **`Dataset configuration must include 'name'`:** you forgot `CachedDatasetMixin` in the class inheritance. The empty `dataset` config dict from the registry routes through a stricter loader that needs more than a string.
- **Class not auto-discovered:** filename must be snake_case and class name PascalCase'd from it. `extract.py` → `class Extract`. Files named `__init__.py`, `run.py`, `base.py`, `metrics.py` are skipped.
- **HF download fails:** check network. `paraloq/json_data_extraction` needs huggingface_hub access. Once cached at `.data/workshop_extraction/workshop_data.jsonl`, subsequent runs are offline.
- **`benchy: command not found`:** use `.venv/bin/benchy` directly, or activate the venv first (`source .venv/bin/activate`).

## Stretch (if time)

- Flip one weight in `metrics_config` (e.g. raise `schema_validity` to `0.8`), re-smoke with a new `--run-id`, and compare. See how sensitive the score is to one knob.
- Add a second subtask to `src/tasks/workshop_extraction/`: create `chat.py` with a `class Chat(CachedDatasetMixin, StructuredHandler)` pointing at a different dataset. Both subtasks run together when you `benchy eval --tasks workshop_extraction`.
