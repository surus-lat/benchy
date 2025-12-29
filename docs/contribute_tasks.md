# Contributing Tasks

This guide is a practical manual for adding a new task to Benchy. The template at
`src/tasks/TASK_TEMPLATE.md` contains full examples, but the sections below explain
how the task system fits together so you can understand the flow without digging
through the codebase.

## Task Basics

A Benchy task is responsible for data, prompts, and evaluation logic. The task loads
or prepares datasets, formats the prompt or request payload, and evaluates model
outputs into per-sample and aggregate metrics. Interfaces handle provider-specific
request formatting and IO, so your task code should remain provider-agnostic.

Internally, tasks implement the `BaseTask` protocol (see `src/engine/protocols.py`).
That protocol defines `load()`, `get_samples()`, `get_prompt()`, `calculate_metrics()`,
`aggregate_metrics()`, and capability properties like `requires_logprobs` and
`requires_multimodal`.

## Step-by-Step

### 1. Copy the Template

```bash
cp -r src/tasks/_template src/tasks/my_task
```

The template includes a `task.json` config, a task class, and a `run.py` wrapper that
connects your task to the shared `TaskGroupRunner`. Use it as a starting point rather
than building from scratch.

### 2. Update `task.json`

Edit `src/tasks/my_task/task.json` to describe your task. This file is the main source
of truth for task metadata and task-level defaults, and it is read by the runner at
runtime.

Key fields to review:
- `name`, `description`, and `output.subdirectory` for naming and output paths.
- `tasks` and `task_configs` if you have subtasks (grouped tasks).
- `defaults` for batch size, retries, timeouts, and logging.
- `capability_requirements` to declare which features are required.

Example capability requirements:

```json
"capability_requirements": {
  "requires_logprobs": "required",
  "requires_multimodal": "none"
}
```

Valid values are `required`, `preferred`, `optional`, or `none`. These feed into the
compatibility checks that block incompatible providers before a run starts.

### 3. Implement the Task Class

In `src/tasks/my_task/task.py`, implement the `BaseTask` methods. The key requirement
is that `get_samples()` yields dictionaries with an `id` and any fields your task uses
(e.g., `text`, `expected`, `schema`). `get_prompt()` should return a system/user pair
for LLM-style interfaces, but HTTP-style interfaces may ignore it and use raw sample
fields instead.

If your task needs to download data, cache it under `src/tasks/my_task/.data/` and
keep preprocessing deterministic so evaluations are reproducible.

### 4. Wire the Runner

In `src/tasks/my_task/run.py`, build a `TaskGroupSpec` and call `run_task_group()`. The
spec tells the runner how to create task instances and aggregate results.

Important `TaskGroupSpec` fields:
- `name` and `display_name`: identifiers used in logs and outputs.
- `default_subtasks`: used when the task is a group and the config omits `tasks`.
- `prepare_task`: a function that builds the task instance per subtask (most tasks use this).
- `run_subtask`: a custom runner if you need non-standard execution (rare).
- `setup` / `teardown`: hooks to load shared resources once and reuse them across subtasks.

The group runner handles building `connection_info`, instantiating the interface, and
running the engine. Your task code should not call providers directly.

### 5. Register the Task

Tasks are discovered via `TASK_REGISTRY` in `src/pipeline.py`. This registry is a map
from task name to metadata that tells the pipeline how to run it. Each entry includes
fields like:
- `run`: the Prefect task function (from your `run.py`).
- `config_name`: the name of the task config to load from `src/tasks/<task>/task.json`.
- `display_name`: human-readable label for logs.
- `set_api_endpoint` and `set_generation_config`: whether to inject extra config.
- `provider_types`: which provider types are allowed for this task.

Add your task name as a key in `TASK_REGISTRY` and make sure the key matches the name
used in configs (e.g., `tasks: ["my_task"]`). If the task is not registered, Benchy
will ignore it even if it appears in configs.

### 6. Leaderboard (Optional)

If this task contributes to leaderboard scoring, update `configs/config.yaml` under
`leaderboard.tasks`. Define a processor, a category score key, and any subcategories
so leaderboard exports include the new task results.

## Validate

Use a small limit to validate the task wiring quickly:

```bash
python eval.py --config configs/tests/spanish-gptoss.yaml --limit 2
```

## Common Pitfalls

- Missing `capability_requirements` can allow incompatible runs that fail later.
- Forgetting to register the task in `TASK_REGISTRY` prevents it from running.
- Task outputs should follow the standard format expected by `save_results()` so
  leaderboard tooling can parse metrics consistently.
