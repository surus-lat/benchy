# Output Artifacts Reference

Benchy writes structured JSON artifacts after every run. These are the authoritative
source of truth for automation, CI pipelines, and result processing scripts.

## Directory layout

```
outputs/
  benchmark_outputs/
    <run_id>/
      <model_name>/
        run_outcome.json        # Run status and per-task results
        run_summary.json        # Compact per-task metric summaries
        <task_group>/
          <subtask>/
            metrics.json        # Task metrics
            samples.jsonl       # Per-sample inputs, outputs, scores
            task_status.json    # Subtask status (used by resume logic)
  probe_outputs/
    <run_id>/
      <model_name>/
        probe_report.json       # Machine-readable capability report
        probe_summary.txt       # Human-readable summary
```

---

## `run_outcome.json`

The machine-facing status source of truth. Written on successful runs, already-completed
runs, and fatal pipeline failures after run directory initialization.

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Artifact schema version (e.g., `"1.0"`) |
| `benchy_version` | string | Benchy package version |
| `model` | string | Model name used in the run |
| `run_id` | string | Run identifier matching the output directory name |
| `timestamp` | string | ISO 8601 timestamp when the run finished |
| `status` | string | Overall run status: `passed`, `degraded`, `failed`, `partial` |
| `exit_policy` | string | Exit policy used: `relaxed`, `smoke`, `strict` |
| `exit_code` | int | Recommended process exit code (0 = success) |
| `started_at` | string | ISO 8601 timestamp when the run started |
| `ended_at` | string | ISO 8601 timestamp when the run ended |
| `duration_s` | float | Total run duration in seconds |
| `git` | object | Git context if available (see below) |
| `counts` | object | Aggregate task counts by status |
| `tasks` | object | Per-task results (keyed by task name) |
| `invocation` | object | Redacted CLI invocation context |
| `artifacts` | object | Indexed paths to output files |
| `errors` | array | Structured error records for fatal failures |
| `diagnostics` | object | Aggregated diagnostic counts across all tasks |

### `git` object

| Field | Type | Description |
|-------|------|-------------|
| `repo` | string | Remote URL of the git repository |
| `commit` | string | Current commit SHA |
| `dirty` | bool | Whether there are uncommitted changes |

### `counts` object

| Field | Type | Description |
|-------|------|-------------|
| `total` | int | Total number of tasks attempted |
| `passed` | int | Tasks with status `passed` |
| `degraded` | int | Tasks with status `degraded` |
| `failed` | int | Tasks with status `failed` |
| `skipped` | int | Tasks with status `skipped` |
| `no_samples` | int | Tasks with status `no_samples` |

### `tasks` object

Keys are task names (e.g., `spanish`, `portuguese.assin2_rte`). Each value:

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Task status (see status vocabulary below) |
| `metrics` | object | Primary task metrics |
| `error_rate` | float | Fraction of samples that errored |
| `invalid_response_rate` | float | Fraction of samples with invalid responses |
| `connectivity_error_rate` | float | Fraction of samples with connectivity errors |
| `total_samples` | int | Total samples attempted |
| `valid_samples` | int | Samples with valid responses |
| `subtask` | string | Subtask name (for single-subtask runs) |
| `artifacts` | object | Paths to task-level output files |

### Task status vocabulary

| Status | Meaning |
|--------|---------|
| `passed` | No connectivity, invalid-response, or error-rate signal |
| `degraded` | Partial issues (`error_rate > 0`, `invalid_response_rate > 0`, or `connectivity_error_rate > 0`) but not total failure |
| `failed` | No valid samples for a non-empty task, all samples failed, or any subtask failed |
| `skipped` | At least one subtask skipped due to capability/requirements mismatch |
| `no_samples` | Task had zero samples or produced no metrics |

### Example

```json
{
  "schema_version": "1.0",
  "benchy_version": "0.1.0",
  "status": "passed",
  "exit_policy": "smoke",
  "exit_code": 0,
  "started_at": "2026-05-28T14:30:12Z",
  "ended_at": "2026-05-28T14:42:07Z",
  "duration_s": 715.4,
  "git": {
    "repo": "https://github.com/surus-lat/benchy.git",
    "commit": "ed98b56",
    "dirty": false
  },
  "counts": {
    "total": 3,
    "passed": 3,
    "degraded": 0,
    "failed": 0,
    "skipped": 0,
    "no_samples": 0
  },
  "tasks": {
    "spanish": {
      "status": "passed",
      "metrics": { "accuracy": 0.74 },
      "error_rate": 0.0,
      "total_samples": 100,
      "valid_samples": 100
    }
  },
  "invocation": {
    "tasks": ["spanish"],
    "limit": null,
    "exit_policy": "smoke"
  },
  "artifacts": {
    "run_summary": "outputs/benchmark_outputs/.../run_summary.json"
  },
  "errors": []
}
```

---

## `run_summary.json`

Compact per-task metric summary. Useful for quick result inspection without parsing
individual task folders.

```json
{
  "run_id": "run_20260528_143012",
  "model": "gpt-4o-mini",
  "timestamp": "2026-05-28T14:42:07",
  "tasks": {
    "spanish": {
      "status": "passed",
      "accuracy": 0.74,
      "total_samples": 100,
      "valid_samples": 100,
      "error_rate": 0.0
    },
    "portuguese": {
      "status": "passed",
      "accuracy": 0.71,
      "total_samples": 80,
      "valid_samples": 80,
      "error_rate": 0.0
    }
  },
  "diagnostics": {}
}
```

---

## `task_status.json`

Written per subtask folder. Used by resume logic when the same `--run-id` is reused to
skip already-completed tasks.

```json
{
  "status": "passed",
  "completed_at": "2026-05-28T14:35:00Z",
  "total_samples": 100,
  "valid_samples": 100
}
```

---

## `probe_report.json`

Machine-readable output of the probe system. Written to
`outputs/probe_outputs/<run_id>/<model>/probe_report.json`.

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Artifact schema version (e.g., `"1.0"`) |
| `benchy_version` | string | Benchy package version |
| `model` | string | Model name |
| `provider_type` | string | Provider name (e.g., `openai`, `vllm`) |
| `base_url` | string | API base URL that was probed |
| `run_id` | string | Run identifier |
| `started_at` | string | ISO 8601 timestamp when the probe started |
| `ended_at` | string | ISO 8601 timestamp when the probe finished |
| `status` | string | Overall probe status: `passed`, `degraded`, `failed` |
| `modes` | object | Request mode results (`chat`, `completions`, `logprobs`) |
| `schema_transports` | object | Structured output support details (see below) |
| `selected_api_endpoint` | string | Chosen request mode for this model (`chat` or `completions`) |
| `selected_schema_transport` | string/null | Chosen schema transport (`structured_outputs`, `response_format`, or null) |
| `api_endpoint_requested` | string | Endpoint mode requested in config (e.g., `auto`) |
| `schema_transport_requested` | string | Schema transport requested in config |
| `schema_transport_forced` | bool | Whether schema transport was forced via `--use-structured-outputs` |
| `checks` | object | Per-check results (`access_readiness`, `multimodal`, `truncation`, `param_support`) |
| `risk_flags` | object | Boolean risk flags (see below) |
| `provider_fingerprint` | object | Server metadata collected during probe |
| `errors` | array | Global errors encountered during probing |
| `test_plan` | object | What checks were run and their pass criteria |
| `known_blindspots` | object | Documented limitations of the probe |

### `schema_transports` object

| Field | Type | Description |
|-------|------|-------------|
| `structured_outputs` | object | Result for `structured_outputs` transport |
| `response_format` | object | Result for `response_format` transport |

Each transport result has a `status` of `ok`, `degraded`, or `failed`.

### `risk_flags` object

| Flag | Type | Meaning |
|------|------|---------|
| `truncation_risk` | bool | Model produces repetition patterns when hitting token limits |
| `repetition_risk` | bool | Model shows degenerate repetition behavior |
| `schema_unreliable` | bool | Structured output accepted by API but quality is unreliable |
| `multimodal_unreliable` | bool | Image inputs may not be reliably supported |

### Example

```json
{
  "schema_version": "1.0",
  "benchy_version": "0.1.0",
  "model": "gpt-4o-mini",
  "provider_type": "openai",
  "base_url": "https://api.openai.com/v1",
  "run_id": "probe_20260528_143000",
  "started_at": "2026-05-28T14:30:00Z",
  "ended_at": "2026-05-28T14:30:42Z",
  "status": "passed",
  "selected_api_endpoint": "chat",
  "selected_schema_transport": "structured_outputs",
  "risk_flags": {
    "truncation_risk": false,
    "schema_unreliable": false,
    "repetition_risk": false,
    "multimodal_unreliable": false
  },
  "schema_transports": {
    "structured_outputs": {"status": "ok"},
    "response_format": {"status": "ok"}
  },
  "checks": {},
  "errors": []
}
```

---

## `metrics.json`

Per-subtask metrics file written inside each task folder. Contents depend on the task
type and handler.

**Classification tasks:**
```json
{
  "accuracy": 0.80,
  "total_samples": 100,
  "valid_samples": 100,
  "error_count": 0,
  "error_rate": 0.0
}
```

**Structured extraction tasks:**
```json
{
  "document_extraction_score": 0.72,
  "extraction_quality_score": 0.68,
  "field_f1_partial": 0.75,
  "schema_validity": 0.95,
  "inverted_hallucination": 0.82,
  "reliable_f1_partial": 0.78,
  "freeform_f1_partial_gt_limited": 0.61,
  "total_samples": 50,
  "valid_samples": 50,
  "error_rate": 0.0
}
```

**Translation tasks:**
```json
{
  "comet": 0.83,
  "chrf": 57.4,
  "bleu": 28.1,
  "total_samples": 100,
  "valid_samples": 100
}
```

---

## `samples.jsonl`

Per-sample detail when `log_samples: true` is set (or `--log-samples` on CLI).
One JSON object per line, containing:

| Field | Description |
|-------|-------------|
| `id` | Sample identifier |
| `input` | Input prompt sent to the model |
| `expected` | Ground-truth expected output |
| `actual` | Model's actual output |
| `score` | Per-sample score (0.0–1.0) |
| `status` | Sample status: `valid`, `error`, `invalid_response` |
| `latency_s` | Request latency in seconds |

---

## Automation recipes

```bash
# Check if a run passed
python -c "import json; d=json.load(open('outputs/benchmark_outputs/my-run/gpt-4o-mini/run_outcome.json')); exit(d['exit_code'])"

# Extract accuracy for spanish task
python -c "import json; d=json.load(open('.../run_summary.json')); print(d['tasks']['spanish']['accuracy'])"

# Watch for run completion
watch -n 5 "cat .../run_outcome.json | python -m json.tool | grep status"
```

---

## Related docs

- [CLI Reference](reference-cli.md) — `--exit-policy` and automation flags
- [Evaluating Models](evaluating_models.md) — Output directory layout
- [AGENTS.md](../AGENTS.md) — Machine-facing automation contract
