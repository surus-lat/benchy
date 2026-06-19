---
name: run-benchmark
description: Run a complete benchmark from a benchmark spec end-to-end. Validates the spec, runs smoke test, then full run, and hands off to read-results. Use when the user has a complete benchmark spec and wants to evaluate their AI system.
---
# Run Benchmark Skill

Runs a benchmark from a spec file end-to-end using the two-stage workflow.

Users never see `--config`, handler class names, or task group identifiers.

When the project has a single spec (or `benchmark.yaml` at the root), commands auto-discover it. When multiple specs exist, always pass `--benchmark <path>` explicitly.

---

## Pre-Flight Check

Before running, validate the spec:

```bash
benchy validate --benchmark <path-to-spec>
```

If validation fails, show the errors and stop:
> "Your benchmark spec has errors that need to be fixed before running:
>   • [error 1]
>   • [error 2]"

Guide the user to fix them using the appropriate skill (define-task, define-scoring, configure-model, setup-data).

---

## Stage 1 — Smoke Run (always first)

```bash
benchy eval --benchmark <path-to-spec> --limit 5 --exit-policy smoke
```

Then read:
```
outputs/benchmark_outputs/<run_id>/<model_name>/run_outcome.json
```

**Proceed to Stage 2 only if ALL of the following hold:**
- process exit code is `0`
- `run_outcome.status` is `passed` or `degraded`
- `run_outcome.counts.failed_tasks == 0`
- `run_outcome.counts.error_tasks == 0`
- `run_outcome.counts.no_samples_tasks == 0`

If smoke fails, diagnose and fix before proceeding:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `no_samples` | Dataset not found | Check `data.path` in the benchmark spec |
| `connectivity_error` | API unreachable | Check `target.url` and network |
| `all_invalid_responses` | Wrong response format | Check `target.response_path` |
| `json_parse_error` | Model not returning JSON | Add a system_prompt to target: |

---

## Stage 2 — Full Run

```bash
benchy eval --benchmark <path-to-spec> --exit-policy strict
```

---

## Output Structure

```
outputs/benchmark_outputs/<run_id>/<model_name>/
├── run_outcome.json       ← authoritative result
├── run_summary.json       ← compact metrics
└── <task>/
    └── <subtask>/
        ├── *_metrics.json
        └── *_samples.json
```

Always parse `run_outcome.json` for pass/fail. Never rely on terminal output.

---

## After a Successful Full Run

Hand off to `read-results`:
> "The benchmark completed. Run `read-results` to see a plain-English summary of the results."

Or invoke it directly by reading `run_outcome.json` and `run_summary.json`.

---

## Re-Running

Same `--run-id` resumes a partial run — already-completed tasks are skipped:
```bash
benchy eval --benchmark <path-to-spec> --run-id my-run --exit-policy strict
```
