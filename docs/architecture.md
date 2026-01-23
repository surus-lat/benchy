# Architecture Overview

Benchy is built around a small set of core concepts: tasks, interfaces, and the engine.
Tasks define *what* to evaluate, interfaces define *how* to talk to a provider, and the
engine ties them together into reproducible benchmark runs. This document walks through
the runtime flow and the major modules so contributors can reason about changes without
hunting through the codebase.

## High-Level Flow

```
benchy eval → ConfigManager → pipeline.py → TaskGroupRunner → BenchmarkRunner → Interface → Provider
```

1. `benchy eval` (`src/benchy_cli_eval.py`) loads the model/system config, sets up logging and run IDs, and starts the
   Prefect flow (`benchmark_pipeline`).
2. `ConfigManager` merges provider configs with model overrides and applies metadata-based
   capability restrictions (`metadata.supports_*`).
3. `pipeline.py` expands task groups, starts/stops vLLM when needed, and dispatches each
   task via its `run_*` wrapper.
4. `TaskGroupRunner` builds connection info, checks compatibility, and instantiates the
   interface and task for each subtask.
5. `BenchmarkRunner` handles batching, retries, and metric aggregation for each task.

## Config Resolution

Model configs live in `configs/models/` and reference provider defaults via
`provider_config`. System configs live in `configs/systems/` and specify a
`provider_type` plus a provider section (for custom endpoints). The config manager
merges provider defaults and overrides so that model-specific settings take precedence
without duplicating every field.

Task groups (like `latam_board`) are defined in `configs/config.yaml`. The pipeline
expands these groups into concrete task names before execution, so the rest of the
system always works with explicit task identifiers.

## Task Execution (TaskGroupRunner)

Each task has a `run.py` wrapper that builds a `TaskGroupSpec` and calls
`run_task_group(...)` in `src/tasks/group_runner.py`. The spec describes how to build
subtasks, how to aggregate metrics, and whether setup/teardown hooks are needed for
shared resources (e.g., loading a metric model once).

`TaskGroupRunner` builds a standardized `connection_info` dict using
`build_connection_info(...)` in `src/engine/connection.py`. That dict is the contract
between the pipeline and interfaces, and includes base URL, API key env var, timeouts,
max tokens, and computed capabilities.

## Benchmark Engine

`BenchmarkRunner` (in `src/engine/benchmark_runner.py`) orchestrates the actual
benchmark loop. It loads samples from the task, batches them, sends requests through
an interface, and aggregates metrics. It also handles checkpointing and can resume
partial runs when checkpoint files are present.

Results are saved via `save_results(...)` and aggregated summaries are produced per
subtask. The pipeline writes a `run_summary.json` at the model output root so you can
see overall completion status in one place.

## Interfaces

Interfaces live in `src/interfaces/` and translate task data into provider requests.
LLM-style interfaces call `task.get_prompt(sample)` and build chat/completions payloads.
HTTP-style interfaces can ignore prompts and use raw fields from the sample.

The engine uses `get_interface_for_provider(...)` to select the correct interface
based on `provider_type`. This keeps task code provider-agnostic and keeps providers
isolated behind a stable API.

## Capabilities and Compatibility

Capabilities are defined in three places and combined at runtime:
- Provider config `capabilities` declare what the serving stack can do.
- Model `metadata.supports_*` tags restrict those capabilities per model.
- Task `capability_requirements` declare required features.

The engine combines provider and model capabilities, then checks them against task
requirements using `check_compatibility(...)` (see `src/engine/protocols.py`). If a
required capability is missing, the run is blocked early with a clear log message.

## Where to Look

- Task system: `src/tasks/`, `src/tasks/group_runner.py`, `src/tasks/TASK_TEMPLATE.md`
- Interfaces: `src/interfaces/`, `src/interfaces/README.md`
- Engine: `src/engine/benchmark_runner.py`, `src/engine/connection.py`, `src/engine/protocols.py`
- Pipeline: `src/pipeline.py`, `src/benchy_cli_eval.py`
