---
name: workshop-define-benchmark
description: Add a new benchy task with custom scoring — copies src/tasks/_template_handler, writes a small Python handler that points at a HuggingFace dataset and declares its own metrics_config, then smoke-runs it. Use when a user wants to define their own structured-extraction benchmark from scratch, especially during the benchy benchmarks workshop.
---

# Workshop — Define a custom benchmark

This skill walks a user through **Part 1** of the benchy benchmarks workshop: adding a new task at `src/tasks/workshop_extraction/` that reuses the production paraloq dataset with a **custom scoring function** (reweighted EQS + stricter string thresholds).

Companion docs: [`.workshop/parts/part1-define-benchmark.md`](../../parts/part1-define-benchmark.md). Production reference: [`.agent/skills/add-task/SKILL.md`](../../../.agent/skills/add-task/SKILL.md).

## When to invoke

Trigger phrases:
- "define a benchmark"
- "add a custom extraction task"
- "score an extraction task my way"
- "Part 1 of the benchy workshop"

## Prerequisites

- Working in the benchy repo root
- `TOGETHER_API_KEY` in `.env`
- At least one model config exists in `configs/models/` (slot A from Part 2 satisfies this)
- benchy installed in `.venv` (`.venv/bin/benchy --help` works)

## Steps

1. **Copy the handler template into a new task directory:**
   ```bash
   cp -r src/tasks/_template_handler src/tasks/workshop_extraction
   cd src/tasks/workshop_extraction
   rm freeform_example.py mcq_example.py README.md
   mv structured_example.py extract.py
   cd ../../..
   ```

2. **Install the workshop handler + metadata** from the pre-staged examples:
   ```bash
   cp .workshop/assets/workshop_extraction_extract.py.example \
      src/tasks/workshop_extraction/extract.py
   cp .workshop/assets/workshop_extraction_metadata.yaml.example \
      src/tasks/workshop_extraction/metadata.yaml
   ```
   The handler uses `CachedDatasetMixin + StructuredHandler`, points at `paraloq/json_data_extraction` (split=train), and declares `metrics_config` with reweighted EQS (schema_validity=0.5, field_f1_partial=0.3, inverted_hallucination=0.2) and stricter string thresholds (exact=0.99, partial=0.80).

3. **Confirm auto-discovery:**
   ```bash
   .venv/bin/benchy tasks | grep workshop_extraction
   ```
   Expected: line `- workshop_extraction` appears in the Tasks list.

4. **Smoke run** (5 samples, forgiving exit policy):
   ```bash
   .venv/bin/benchy eval \
     --config configs/models/together_modelA.yaml \
     --tasks workshop_extraction \
     --limit 5 --exit-policy smoke --run-id w1_smoke
   ```
   Wall-clock: ~30-60s.

5. **Inspect:**
   ```bash
   cat outputs/benchmark_outputs/w1_smoke_LIMITED/<model>/run_outcome.json \
     | jq '{status, task: .tasks.workshop_extraction.status, summary: .tasks.workshop_extraction.summary}'
   cat outputs/benchmark_outputs/w1_smoke_LIMITED/<model>/run_summary.json \
     | jq '.tasks.workshop_extraction.extraction_quality_score'
   ```
   Expected: `status: "passed"` (or "degraded" with non-zero EQS).

6. **(Comparison)** Run the production paraloq task on the same model with the same limit:
   ```bash
   .venv/bin/benchy eval \
     --config configs/models/together_modelA.yaml \
     --tasks structured_extraction \
     --limit 5 --exit-policy smoke --run-id w1_smoke_prod
   ```
   Then compare `extraction_quality_score` in the two `run_summary.json` files. Same data, different weights → different scores. That's the lesson.

## Captured outputs

Report back to the user:
- Paths of the new files (`src/tasks/workshop_extraction/extract.py`, `.../metadata.yaml`)
- The smoke `run_id` (`w1_smoke_LIMITED` after benchy appends the suffix)
- `run_outcome.status` and the headline EQS from `run_summary.json`
- Side-by-side EQS scores: workshop weights vs production weights

## Recovery hints

- **`Dataset configuration must include 'name'`:** the class is missing `CachedDatasetMixin` or `_download_and_cache`. The bundled example has both. If you wrote your own class, add the mixin.
- **Class not auto-discovered:** filename must be snake_case and the class name PascalCase'd from it. `extract.py` → `class Extract`. Files named `__init__.py`, `run.py`, `base.py`, `metrics.py` are skipped by the registry.
- **HF download fails:** check network. `paraloq/json_data_extraction` requires huggingface_hub access. Once cached at `.data/workshop_extraction/workshop_data.jsonl`, subsequent runs are offline.
- **`benchy: command not found`:** use `.venv/bin/benchy` directly, or activate the venv first (`source .venv/bin/activate`). benchy ships in the project venv, not globally.

## Why this beats the wizard route (instructor cue)

The `.agent/skills/{define-task,define-scoring,setup-data,configure-model}` wizard skills exist but write a `benchmark.yaml` spec that the current CLI doesn't consume — `benchy validate` and `benchy eval --benchmark <path>` are not implemented. The Python handler route is what production uses and what's actually testable end-to-end today.
