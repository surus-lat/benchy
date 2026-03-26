---
name: add-task
description: Add a new benchmark task or task group to benchy. Covers handler selection, directory layout, metadata.yaml, task class implementation, __init__.py exports, and smoke-test verification. Use when asked to add a new evaluation task, benchmark, or task group.
---
# Add Task Skill

Benchy uses **handler-based tasks**: inherit from a format handler, declare config, override only what you need. Most tasks are 30–50 lines of code.

---

## Step 0 — Choose Your Handler

| Task format | Handler | Use when |
|---|---|---|
| Multiple choice A/B/C/D | `MultipleChoiceHandler` | Fixed or per-sample choices |
| JSON extraction | `StructuredHandler` | Structured data from text |
| Open-ended generation | `FreeformHandler` | Custom metrics, free text |
| Image + structured output | `MultimodalStructuredHandler` | Document/image extraction |

---

## Step 1 — Copy the Template

```bash
cp -r src/tasks/_template_handler src/tasks/<my_benchmark>
```

The template contains annotated examples for each handler type plus `metadata.yaml`.

---

## Step 2 — Create `metadata.yaml`

```yaml
# src/tasks/<my_benchmark>/metadata.yaml
name: my_benchmark
display_name: My Benchmark Suite
description: One-line description of what this suite evaluates

capability_requirements:
  requires_logprobs: optional    # required | preferred | optional | none
  requires_multimodal: none
  requires_schema: optional
  requires_files: none

subtasks:
  my_subtask:
    description: What this subtask evaluates
    dataset_url: https://huggingface.co/datasets/org/name
```

---

## Step 3 — Implement Task Classes

### Multiple Choice

```python
# src/tasks/my_benchmark/my_task.py
from ..common import CachedDatasetMixin, MultipleChoiceHandler

class MyTask(CachedDatasetMixin, MultipleChoiceHandler):
    name = "my_task"
    display_name = "My Task"

    dataset_name = "org/dataset"
    split = "test"
    dataset_file = "my_task.jsonl"

    labels = {0: "No", 1: "Yes"}
    system_prompt = "Answer the question."

    def _download_and_cache(self, output_path):
        from ..common import download_huggingface_dataset, save_to_jsonl
        raw = download_huggingface_dataset(self.dataset_name, self.split)
        processed = [
            {"id": f"item_{i}", "text": s["text"], "expected": s["label"]}
            for i, s in enumerate(raw)
        ]
        save_to_jsonl(processed, output_path)
```

### Structured Extraction

```python
from ..common import CachedDatasetMixin, StructuredHandler

class MyExtraction(CachedDatasetMixin, StructuredHandler):
    name = "my_extraction"
    display_name = "My Extraction"

    dataset_name = "org/dataset"
    split = "test"
    dataset_file = "my_extraction.jsonl"

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "date": {"type": "string"},
        },
        "required": ["name", "date"],
        "additionalProperties": False,
    }
    system_prompt = "Extract structured data from the text."
```

### Multimodal Structured (images/documents)

```python
from ..common import MultimodalStructuredHandler

class MyVisionTask(MultimodalStructuredHandler):
    name = "my_vision"
    display_name = "My Vision Task"

    requires_multimodal = True
    requires_files = True
    input_type = "image"
    image_field = "image_path"
    schema_field = "schema"

    system_prompt = "Extract data from this document image."

    def load_dataset(self):
        # Return list of dicts with image_path, schema, expected
        ...
```

### Freeform with Custom Metrics

```python
from ..common import FreeformHandler, ExactMatch, F1Score

class MySummary(FreeformHandler):
    name = "my_summary"
    display_name = "My Summary"

    def __init__(self, config=None):
        super().__init__(config)
        self.metrics = [ExactMatch(), F1Score()]

    def load_dataset(self):
        return [{"id": "1", "text": "...", "expected": "..."}]

    def get_prompt(self, sample):
        return "You are a summarizer.", f"Summarize:\n\n{sample['text']}"
```

---

## Step 4 — Export `__init__.py`

```python
# src/tasks/<my_benchmark>/__init__.py
"""My benchmark task group."""

from .my_task import MyTask
from .my_extraction import MyExtraction

__all__ = ["MyTask", "MyExtraction"]
```

---

## Step 5 — Register in `configs/config.yaml` (optional)

To add the task group to a named group (e.g., `latam_board`):

```yaml
# configs/config.yaml
task_groups:
  my_group:
    - my_benchmark.my_task
    - my_benchmark.my_extraction
```

---

## Step 6 — Smoke Test

```bash
benchy eval --config configs/models/openai_gpt-4o-mini.yaml \
    --tasks my_benchmark.my_task --limit 5 --exit-policy smoke
```

---

## Data Format Reference

### Required fields per sample

| Handler | Required fields |
|---|---|
| `MultipleChoiceHandler` | `id`, `text`, `expected` (int index) |
| `StructuredHandler` | `id`, `text`, `schema` (JSON), `expected` (dict) |
| `MultimodalStructuredHandler` | `id`, `image_path`, `schema`, `expected` |
| `FreeformHandler` | `id`, `text`, `expected` |

### Per-sample choices (MCQ)

When each sample has different choices:
```python
{
    "id": "1",
    "text": "Question?",
    "choices": ["Option A", "Option B"],
    "choice_labels": ["A", "B"],
    "expected": 0   # index into choices
}
```

---

## Directory Layout

```
src/tasks/<my_benchmark>/
├── metadata.yaml          ← task group declaration (required)
├── __init__.py            ← exports task classes (required)
├── my_task.py             ← one file per subtask
├── my_extraction.py
└── .data/                 ← auto-created, cached dataset files
    └── my_task.jsonl
```

---

## Useful Utilities

```python
from src.tasks.common import (
    # Dataset I/O
    download_huggingface_dataset,
    save_to_jsonl, load_jsonl,
    load_tsv, load_csv,
    # Text
    normalize_text, normalize_spaces, remove_accents,
    format_choices, extract_choice_letter,
    build_fewshot_block,
    # Metrics
    ExactMatch, F1Score,
    MultipleChoiceAccuracy,
    MeanSquaredError, PearsonCorrelation,
)
```

---

## Capability Declarations

Set in `metadata.yaml`. Valid values: `required`, `preferred`, `optional`, `none`.

```yaml
capability_requirements:
  requires_logprobs: preferred    # MCQ benefits from logprobs scoring
  requires_multimodal: required   # Must have image support
  requires_schema: preferred      # Structured output enforcement
  requires_files: required        # File upload capability
```

Tasks are automatically skipped when required capabilities are missing.

---

## Metrics Config (Structured Tasks)

Control partial-match thresholds:

```python
metrics_config = {
    "partial_matching": {
        "string": {
            "exact_threshold": 0.95,
            "partial_threshold": 0.50,
        }
    }
}
```

For OCR/image tasks, lower thresholds (0.85 / 0.40) are more appropriate.

---

## Ground Truth Examples

- `src/tasks/spanish/` — MCQ with Spanish text
- `src/tasks/portuguese/` — MCQ + regression
- `src/tasks/structured_extraction/` — JSON extraction from text
- `src/tasks/document_extraction/` — Document extraction from images
- `src/tasks/_template_handler/README.md` — Annotated walkthrough
