```skill
---
name: interpret-run
description: Read and interpret benchy run outputs. Covers run_outcome.json structure, status vocabulary, metric summaries, failure diagnosis, per-task/subtask breakdown, and actionable next steps. Use when asked to analyze, explain, or act on benchy evaluation results.
---
# Interpret Run Skill

Always parse JSON artifacts — never parse human logs. `run_outcome.json` is the source of truth.

---

## Output Location

```
outputs/benchmark_outputs/<run_id>/<model_name_last_segment>/
├── run_outcome.json      ← authoritative status
├── run_summary.json      ← compact metric table
└── <task>/
    ├── task_status.json  ← resume checkpoint
    └── <subtask>/
        ├── *_metrics.json
        ├── *_samples.json
        └── *_report.txt
```

`model_name_last_segment` = last path component of the model name (e.g. `gpt-4o-mini`).

---

## `run_outcome.json` Top-Level Keys

```json
{
  "schema_version": 1,
  "benchy_version": "...",
  "model": "org/model-name",
  "run_id": "my-run-2024",
  "status": "passed",
  "exit_policy": "strict",
  "exit_code": 0,
  "started_at": "...",
  "ended_at": "...",
  "duration_s": 42.3,
  "counts": {
    "total_tasks": 3,
    "passed_tasks": 3,
    "degraded_tasks": 0,
    "failed_tasks": 0,
    "skipped_tasks": 0,
    "error_tasks": 0,
    "no_samples_tasks": 0,
    "pending_tasks": 0
  },
  "tasks": { ... },
  "invocation": { "argv": [...], "cli_args": {...}, "cwd": "..." },
  "git": { "repo": "...", "commit": "...", "dirty": false },
  "artifacts": { ... },
  "errors": []
}
```

---

## Status Vocabulary

| Status | Meaning | Action |
|---|---|---|
| `passed` | All samples processed, no errors | ✅ Proceed |
| `degraded` | Partial errors (some samples failed) | ⚠️ Review error rate |
| `failed` | All samples failed or no valid responses | ❌ Fix before proceeding |
| `skipped` | Task skipped due to incompatibility | Check capability mismatch |
| `no_samples` | Dataset empty or not found | Fix dataset path/download |
| `error` | Pipeline-level exception | Check `errors` field |
| `pending` | Task listed but never ran | Indicates interrupted run |

---

## Overall Run Status Logic

```
Any task failed/error/pending  →  run status: failed
Any task degraded/skipped/no_samples  →  run status: degraded
All tasks passed  →  run status: passed
```

---

## Agent Decision Rule (from AGENTS.md)

**Proceed to full run only if ALL of these hold:**

```python
exit_code == 0
run_outcome["status"] in {"passed", "degraded"}
run_outcome["counts"]["failed_tasks"] == 0
run_outcome["counts"]["error_tasks"] == 0
run_outcome["counts"]["pending_tasks"] == 0
run_outcome["counts"]["no_samples_tasks"] == 0
run_outcome["counts"]["skipped_tasks"] == 0
```

---

## Per-Task Breakdown

Each entry in `run_outcome["tasks"]`:

```json
{
  "status": "degraded",
  "reason": "subtask_degraded",
  "summary": {
    "total_samples": 100,
    "valid_samples": 92,
    "error_count": 8,
    "error_rate": 0.08,
    "invalid_response_rate": 0.05,
    "connectivity_error_rate": 0.03,
    "response_rate": 0.92
  },
  "subtasks": {
    "my_subtask": {
      "status": "degraded",
      "reason": "partial_errors",
      "summary": { ... }
    }
  }
}
```

### `reason` values

| Reason | Meaning |
|---|---|
| `subtask_failure` | At least one subtask failed |
| `subtask_skipped` | At least one subtask was skipped |
| `subtask_no_samples` | At least one subtask had no data |
| `subtask_degraded` | At least one subtask had partial errors |
| `partial_errors` | Some samples failed but most succeeded |
| `all_connectivity_errors` | 100% connectivity failures |
| `all_invalid_responses` | 100% unparseable/wrong-format responses |
| `no_valid_samples` | No samples produced valid output |
| `all_samples_failed` | Error rate = 100% |
| `no_samples` | Dataset empty |
| `incompatible` | Task skipped due to capability mismatch |

---

## Metric Interpretation

### Key metrics in `*_metrics.json`

| Metric | Healthy range | Concern |
|---|---|---|
| `error_rate` | 0.0 | > 0.10 is high |
| `connectivity_error_rate` | 0.0 | Any > 0 means network/auth issues |
| `invalid_response_rate` | 0.0 | > 0.05 means prompt/format issue |
| `response_rate` | 1.0 | < 0.90 warrants investigation |
| `exact_match_rate` | task-dependent | Compare to baseline |
| `partial_match_rate` | task-dependent | Should be ≥ exact_match_rate |
| `field_f1_strict` | 0.0–1.0 | Structured extraction quality |
| `eqs` (Extraction Quality Score) | 0.0–1.0 | Composite extraction metric |
| `accuracy` | 0.0–1.0 | MCQ classification accuracy |

---

## Failure Diagnosis Workflow

### 1. Read `run_outcome["counts"]`
Identify which count is non-zero.

### 2. Look at `run_outcome["tasks"]` 
Find the failing task(s), read their `reason` and `summary`.

### 3. Read `run_outcome["errors"]`
Structured error records with type, message, traceback.

### 4. Check per-subtask samples
For connectivity errors: `*_samples.json` → samples with `error_type: "connectivity"`.
For invalid responses: samples with `error_type: "invalid_response"` show the raw model output.

### 5. Common fixes

| `reason` | Fix |
|---|---|
| `all_connectivity_errors` | Check API key env var, base URL, network connectivity |
| `all_invalid_responses` | Check model name, api_endpoint mode (chat vs completions); **for multimodal tasks: check image dimensions** — add `image_max_edge` to provider config if endpoint has a px limit |
| `no_samples` | Check dataset path, run download script |
| `incompatible` | Check `capability_requirements` in `metadata.yaml` vs provider capabilities |
| `subtask_no_samples` | One subtask missing data; check `.data/` directory |

---

## `run_summary.json`

Compact table of metrics per subtask. Useful for quick comparison across runs.

```json
{
  "model": "...",
  "run_id": "...",
  "tasks": {
    "my_task.subtask": {
      "accuracy": 0.85,
      "total_samples": 100,
      "error_rate": 0.02
    }
  }
}
```

---

## Git Metadata

```json
"git": {
  "repo": "org/benchy",
  "commit": "abc1234",
  "dirty": false
}
```

`dirty: true` means local uncommitted changes — important for reproducibility.

---

## Reading Samples for Debugging

When a task is `degraded` or `failed`, read the samples file.

**Important:** samples files are wrapped dicts `{"model":..., "task":..., "samples":[...]}`, not bare lists:

```python
import json
with open("outputs/benchmark_outputs/<run_id>/<model>/<task>/<subtask>/*_samples.json") as f:
    data = json.load(f)

# Unwrap the envelope
all_samples = data["samples"] if isinstance(data, dict) else data

# Find failures
failures = [s for s in all_samples if s.get("error") or s.get("error_type")]
# Find wrong predictions
wrong = [s for s in all_samples if s.get("prediction") != s.get("expected")]
```

---

## Resume a Failed Run

If the run was interrupted or you fixed an issue:

```bash
benchy eval --config <config> --run-id <same-run-id> --exit-policy strict
```

Tasks already marked `passed` or `degraded` in `task_status.json` are skipped.
```
