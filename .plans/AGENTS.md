# Benchy Agent Contract

This file defines the machine-facing contract for running Benchy from automated agents.

## Canonical Entrypoint

Use `benchy eval`.

## Canonical Workflow

1. Smoke run:

```bash
benchy eval --config <config-or-name> --tasks <task...> --limit 5 --run-id <id> --exit-policy smoke
```

2. Read `<model_output_dir>/run_outcome.json`.
3. Continue to full run only if:
- process exit code is `0`
- `run_outcome.status` is `passed` or `degraded`
- `run_outcome.counts.failed_tasks == 0`
- `run_outcome.counts.error_tasks == 0`
- `run_outcome.counts.pending_tasks == 0`
- `run_outcome.counts.no_samples_tasks == 0`
- `run_outcome.counts.skipped_tasks == 0`

4. Full run:

```bash
benchy eval --config <config-or-name> --tasks <task...> --run-id <id> --exit-policy strict
```

## Output Location Contract

Run outputs are written to:

`<base_output_path>/<run_id>/<model_name_last_segment>/`

Required files:
- `run_outcome.json` (status source of truth)
- `run_summary.json` (metric summary)
- `<task>/task_status.json` (resume status per task group)

`run_outcome.json` is written on successful runs, already-completed runs, and fatal
pipeline failures after run directory initialization.

## `run_outcome.json` Contract

Top-level keys:
- `schema_version`
- `benchy_version`
- `model`
- `run_id`
- `status`
- `exit_policy`
- `exit_code`
- `started_at`
- `ended_at`
- `duration_s`
- `counts`
- `tasks`
- `invocation`
- `git`
- `artifacts`
- `errors`

Status vocabulary:
- `passed`
- `degraded`
- `failed`
- `skipped`
- `no_samples`
- `error`
- `pending`

## Exit Policy Contract

- `relaxed`: returns `0` unless fatal exception.
- `smoke`: returns non-zero if any task is `failed`, `error`, `pending`, `skipped`, or `no_samples`.
- `strict`: returns non-zero unless every requested task is `passed`.

## Generic API Benchmarking

Agents can benchmark arbitrary HTTP API endpoints using the `--api-url` flag.
This is the preferred way to evaluate pipelines or custom endpoints without writing code.

### Required flags

| Flag | Description |
|------|-------------|
| `--api-url <url>` | Target endpoint URL (sets provider to `api`) |
| `--api-body-template <json>` | JSON template with `{{field}}` placeholders from dataset samples |
| `--model-name <label>` | Label for the system under test (used in output artifacts) |
| `--tasks <task...>` | Task(s) providing dataset and metrics |

### Optional flags

| Flag | Description |
|------|-------------|
| `--api-response-path <path>` | Dot-notation path to extract output (e.g. `data`, `choices.0.message.content`) |
| `--api-method <method>` | HTTP method (default: `POST`) |
| `--api-headers <json>` | Extra headers as JSON object |
| `--api-key-env <var>` | Environment variable holding auth token (default: `API_KEY`) |
| `--api-key <key>` | Direct API key value |

### Template placeholders

- `{{field}}` — string substitution from dataset sample
- `{{field|base64_image}}` — image file path → base64 data URL
- `{{field|json}}` — embed native JSON (preserves dicts/lists)

### Canonical API benchmark workflow

```bash
# 1. Smoke run
benchy eval \
  --api-url "https://api.example.com/extract" \
  --api-key-env MY_API_KEY \
  --api-body-template '{"image": "{{image_path|base64_image}}"}' \
  --api-response-path "data" \
  --tasks document_extraction.facturas_argentinas \
  --model-name "my-pipeline-v1" \
  --limit 5 --run-id smoke_api --exit-policy smoke

# 2. Read run_outcome.json, verify passed/degraded

# 3. Full run
benchy eval \
  --api-url "https://api.example.com/extract" \
  --api-key-env MY_API_KEY \
  --api-body-template '{"image": "{{image_path|base64_image}}"}' \
  --api-response-path "data" \
  --tasks document_extraction.facturas_argentinas \
  --model-name "my-pipeline-v1" \
  --run-id full_api --exit-policy strict
```

All existing flags (`--limit`, `--exit-policy`, `--run-id`, `--image-max-edge`, etc.) work with API mode. The same `run_outcome.json` and `run_summary.json` contracts apply.

## Zero-Code Dataset Evaluation

Agents can evaluate any dataset in `.data/` without writing Python. Datasets are auto-discovered with full metadata inference.

### Evaluate a `.data/` dataset

```bash
# 1. Discover available datasets
benchy datasets --json

# 2. Smoke test
benchy eval --dataset-name <name> --task-type <structured|classification|freeform> \
  --provider openai --model-name gpt-4o --limit 3 --run-id <id>_smoke --exit-policy smoke

# 3. Read run_outcome.json, verify passed
# 4. Full run
benchy eval --dataset-name <name> --task-type <type> \
  --provider openai --model-name gpt-4o --run-id <id>_full --exit-policy strict
```

### Custom prompts

Override prompts per-dataset without modifying any files:

```bash
benchy eval --dataset-name <name> --task-type <type> \
  --provider openai --model-name gpt-4o \
  --system-prompt "Your custom system instruction." \
  --user-prompt-template "Your template with {text} and {schema} placeholders." \
  --limit 5
```

For classification, available placeholders: `{text}`, `{choices}`.
For extraction, available placeholders: `{text}`, `{schema}`.

### After adapting a new dataset

After placing a new dataset in `.data/` and validating it works:

1. Write a `benchy.md` in the dataset directory with smoke test, full run, and custom prompt commands
2. Verify the smoke test passes: `benchy eval --dataset-name <name> --task-type <type> --provider openai --model-name gpt-4o --limit 3`
3. Follow `docs/DATASET_SPEC.md` for required files and column conventions

## Agent Rules

- Parse JSON artifacts, not human logs.
- Treat `run_outcome.json` as authoritative for run success/failure.
- Reusing the same `run_id` must skip tasks already marked completed in `<task>/task_status.json`.
- Samples files are wrapped dicts `{"model":..., "samples":[...]}` — access `data["samples"]`, not the top-level object.
- For multimodal HTTP endpoints, always verify `image_max_edge` is set. `all_invalid_responses` from image tasks often means the API pixel-size limit was exceeded (not a model error).
- `document_extraction` → use `configs/systems/surus-factura.yaml`; requires `image_max_edge: 2048` (SURUS API hard limit: 2560px).

---

## Available Skills

Skills are loaded on demand — only the short description stays in context.
Invoke a skill when you need step-by-step guidance for these workflows:

| Skill | Description |
|---|---|
| `evaluate` | Run benchy evals (smoke→full workflow, config selection, exit policies) |
| `add-task` | Add a new benchmark task or task group |
| `add-provider` | Add a new inference provider (OpenAI-compatible or custom HTTP) |
| `interpret-run` | Read and diagnose run_outcome.json, metrics, and failure patterns |

Skill files: `skills/<name>/SKILL.md`

---

## MCP Server

The benchy MCP server exposes run outputs and config as tools for agent use.

Install:
```bash
pip install benchy[mcp]
```

Start:
```bash
benchy-mcp
# or with custom output path:
benchy-mcp --output-path /my/outputs
```

Available tools:
- `read_run_outcome(run_id, model_name?)` — parse `run_outcome.json`
- `validate_smoke_gates(run_id, model_name?)` — check AGENTS.md smoke contract
- `list_runs(limit?)` — recent runs with status summary
- `list_configs(kind?)` — available model/system configs
- `list_tasks(group?)` — available tasks from `reference/tasks_list.json`
- `read_run_summary(run_id, model_name?)` — compact metric table
- `get_task_errors(run_id, task_name, model_name?, max_samples?)` — failed samples

Server source: `src/mcp/server.py`

---

## Validation Script

Validate a run against smoke gates from the command line or CI:

```bash
python scripts/validate_run.py --run-id <id>
# exits 0 if gates pass, 1 if not; prints structured JSON
```

Wire into a workflow:
```bash
benchy eval --config ... --run-id my-run --limit 5 --exit-policy smoke \
  && python scripts/validate_run.py --run-id my-run
```
