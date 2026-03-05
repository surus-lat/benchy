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

## Agent Rules

- Parse JSON artifacts, not human logs.
- Treat `run_outcome.json` as authoritative for run success/failure.
- Reusing the same `run_id` must skip tasks already marked completed in `<task>/task_status.json`.

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
