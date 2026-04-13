# New User Benchmark Creation — Gap Analysis

A new user wants to define a benchmark by saying: "this is my scoring function, this is my data, these are my inputs and outputs." This document maps the current journey and identifies where abstractions and steps are missing or unclear.

---

## Current Journey (9 Steps)

**Step 1 — Pick a handler.** A table in SKILL.md and contribute_tasks.md describes the 4 handlers (MCQ, Structured, Freeform, Multimodal). Gap: no decision tree — the user must read to decide. "Per-sample choices" mode (MCQ where options vary per question) is buried in a docstring, not surfaced as a first-class feature.

**Step 2 — Copy template & look at examples.** `src/tasks/_template_handler/` has 3 annotated templates + README; real examples live in `src/tasks/spanish/`, `src/tasks/document_extraction/`. Gap: the template shows all features at once. There is no tiered path — "simple (5 lines)" vs. "advanced (50 lines)" — so new users don't know what's required vs. optional.

**Step 3 — Write `metadata.yaml`.** A template exists in SKILL.md alongside real examples. Gap: real tasks use `group`, `group_metadata`, `output.subdirectory`, `metrics_manifest` — none of these appear in the docs. Capability values (`required/preferred/optional/none`) are listed but what actually happens when a model lacks that capability is never explained.

**Step 4 — Define the task class.** Code examples exist for all 4 handlers. `CachedDatasetMixin` handles HF download + caching. Gap: required sample fields (`id`, `text`, `expected`, etc.) are documented in 3 places with slight differences — no single source of truth. `CachedDatasetMixin` requires understanding caching mechanics; alternative loading patterns (local file, custom API) are not shown. No sample validation — bad data silently causes wrong metrics.

**Step 5 — Define metrics.** MCQ is automatic. Structured is automatic (EQS, field-level F1, hallucination rate), configurable via a `metrics_config` dict. Freeform requires `self.metrics = [ExactMatch(), F1Score()]` in `__init__()`. Gap: two incompatible configuration patterns (dict for structured, list for freeform, nothing for MCQ). The `ScalarMetric` protocol for custom metrics exists in code but is not documented in any task guide. No unified metrics reference.

**Step 6 — Export in `__init__.py`.** 3 lines of boilerplate. No gaps.

**Step 7 — Register in task group (optional).** Documented in SKILL.md. No gaps.

**Step 8 — Smoke test.** Exact command given in SKILL.md (`--limit 5 --exit-policy smoke`). Gap: no guidance on what "healthy" smoke output looks like. No debugging guide — if it fails, how do you trace the error? Output directory structure is not explained in task-creation docs. `--exit-policy` semantics are not defined inline.

**Step 9 — Full run.** CLI flags and config files exist with examples. Gap: config vs. flag precedence is undocumented (which wins?). 50+ flags; only a handful are shown in examples.

---

## Missing Abstractions (Highest Signal)

**1. No unified entry point for "I have data + scoring function."** Today a user must pick a handler, understand mixins, write Python, and wire up metadata. There is no zero-code path for a user who just has a JSONL + a scoring function and doesn't want to write a handler class.

**2. Sample format has no formal schema.** Required fields per handler are documented in prose across 3 files. There is no TypedDict/Pydantic model, no `benchy validate` command to check before running, and no helpful error when a field is missing.

**3. Metrics configuration has two incompatible patterns.** Structured uses a dict (`metrics_config`), Freeform uses a list in `__init__()`, MCQ has nothing. A user defining a custom scoring function has no clear interface to implement.

**4. No feedback loop during task development.** The smoke test is the only check. There is no `benchy validate --task` to pre-flight dataset format, metadata, and metrics config. Failures surface only after a full pipeline run.

**5. Documentation is fragmented across 4 files.** SKILL.md, contribute_tasks.md, HANDLER_SYSTEM_GUIDE.md, DATASET_SPEC.md all partially overlap. There is no single "task development start here" page that routes a user through the right docs in order.

---

## What Would Close the Biggest Gap

The highest-value intervention is a **zero-code path** for the "I have data + scoring function" case:

```bash
benchy eval \
  --dataset-name my_data \
  --task-type freeform \
  --scoring exact_match \
  --model-name gpt-4o \
  --limit 5 --exit-policy smoke
```

This partially exists (`--dataset-name`, `--task-type`) but the `--scoring` abstraction is missing — users still must write Python to define anything beyond the defaults.

Second highest: `benchy validate --task <name>` pre-flight check.
Third: a single `docs/TASK_DEVELOPMENT.md` that routes users through steps 1–9 in one place.
