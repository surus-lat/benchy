---
name: evaluate
description: Run benchy evaluations against models or systems. Covers the canonical smoke→full workflow, config selection, task filtering, exit policies, and reading run_outcome.json. Use when asked to evaluate, benchmark, or run benchy against a model or system config.
---
# Evaluate Skill

Run Benchy evaluations reliably using the canonical two-stage workflow defined in `AGENTS.md`.

## Canonical Workflow

### Stage 1 — Smoke Run (always first)

```bash
benchy eval --config <config-or-name> --tasks <task...> --limit 5 --run-id <id> --exit-policy smoke
```

Then read:

```
outputs/benchmark_outputs/<run_id>/<model_name_last_segment>/run_outcome.json
```

**Proceed to Stage 2 only if ALL of the following hold:**
- process exit code is `0`
- `run_outcome.status` is `passed` or `degraded`
- `run_outcome.counts.failed_tasks == 0`
- `run_outcome.counts.error_tasks == 0`
- `run_outcome.counts.pending_tasks == 0`
- `run_outcome.counts.no_samples_tasks == 0`
- `run_outcome.counts.skipped_tasks == 0`

If smoke fails: read `errors` in `run_outcome.json`, check provider connectivity, fix config, re-run smoke. Do not proceed to full run.

> **Before adding new tasks or providers**, check the `add-task` and `add-provider` skills. They cover task handler implementation, interface wiring, and config architecture in detail. This skill focuses on running evaluations.

### Stage 2 — Full Run

```bash
benchy eval --config <config-or-name> --tasks <task...> --run-id <id> --exit-policy strict
```

---

## Config Selection

| Scenario | Config location |
|---|---|
| General model (vLLM, OpenAI, Anthropic...) | `configs/models/` |
| Task-specific system (SURUS, custom HTTP) | `configs/systems/` |
| Smoke test preset | `configs/tests/` |

Config can be a full path or just a name — benchy searches `configs/models/`, `configs/systems/`, `configs/tests/`, and `configs/`.

**Cloud API example:**
```bash
benchy eval --config configs/models/openai_gpt-4o-mini.yaml --tasks spanish --limit 5
```

**Local vLLM example:**
```bash
benchy eval --model-path /path/to/model --model-name my-model \
    --vllm-config vllm_two_cards_mm --tasks latam_board --limit 5
```

**System endpoint example:**
```bash
benchy eval --config configs/systems/surus-factura.yaml --tasks document_extraction.facturas_argentinas --limit 5
```

---

## Task Selection

- `--tasks a b c` or `--tasks a,b,c` — explicit task list, overrides config
- `--task-group latam_board` — expands a named group from `configs/config.yaml`
- `--tasks-file path.txt` — one task per line, `#` comments allowed
- No `--tasks` flag → uses config's built-in task list

For system providers, `--tasks` is intersected against the config's declared tasks (cannot run unsupported tasks).

---

## Exit Policies

| Policy | When to use | Exits non-zero when |
|---|---|---|
| `relaxed` | local dev, interactive | never (unless crash) |
| `smoke` | pre-flight check (stage 1) | any task failed/skipped/error/no_samples |
| `strict` | CI/production (stage 2) | any task is not `passed` |

---

## Useful Flags

```
--limit N          Samples per task (fast smoke tests)
--run-id NAME      Custom run folder name (enables resume)
--log-samples      Force sample logging for all tasks
--batch-size N     Override default batch size
--image-max-edge N Downscale images to N px max edge (multimodal)
--compatibility    warn|skip|error — how to handle incompatible tasks
--output-path      Override output base dir (or 'model' for side-by-side)
--save-config FILE Persist CLI params as reusable YAML
```

---

## Output Structure

```
outputs/benchmark_outputs/<run_id>/<model_name>/
├── run_outcome.json      ← authoritative status (parse this, not logs)
├── run_summary.json      ← compact metric summary
└── <task>/
    ├── task_status.json  ← resume checkpoint
    ├── <subtask>/
    │   ├── *_metrics.json
    │   ├── *_samples.json
    │   └── *_report.txt
```

Always parse `run_outcome.json`. Never parse human logs for pass/fail decisions.

---

## Environment Setup

Copy `env.example` to `.env` and set API keys:

```bash
cp env.example .env
# Edit .env: add OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
```

---

## Common Failure Patterns

| Symptom | Likely cause | Fix |
|---|---|---|
| `no_samples` | Dataset not found or path wrong | Check dataset path and download |
| `skipped` | Capability mismatch | Check `capability_requirements` vs provider capabilities |
| `connectivity_error` | API unreachable | Check base_url, API key, network |
| `all_invalid_responses` | Wrong model/endpoint or oversized images | Verify model name, endpoint mode; for multimodal endpoints add `image_max_edge` to config |
| `valid_json_non_schema` | Model returned valid JSON but wrong schema | Check what the API is returning — the output format may not match the expected schema |
| `degenerate_repetition` | Model output degenerated into repetition | Prompt issue or model instability — check `raw_length` and `diagnostics.raw` in samples |
| exit code 1 + smoke policy | Any task not clean | Read `run_outcome.json` errors section and `diagnostic_class` in samples |

---

## Resume a Partial Run

Reuse the same `--run-id` — tasks already marked completed in `task_status.json` are skipped automatically.

```bash
benchy eval --config my-config.yaml --run-id my-run-id --exit-policy strict
```

---

## Diagnosing Smoke Failures

### Read sample diagnostics

Each samples file has a `diagnostics` block per sample:

```python
import json
with open("outputs/benchmark_outputs/<run_id>/<model>/<task>/<task>_samples.json") as f:
    data = json.load(f)
for s in data["samples"][:3]:
    print(s["id"], s["diagnostics"]["diagnostic_class"])
    print("  raw:", s.get("raw_prediction", "")[:200])
    print("  expected keys:", list(s.get("expected", {}).keys())[:5])
```

**Key diagnostic classes:**

| `diagnostic_class` | Meaning | Next step |
|---|---|---|
| `valid_json_schema` | Clean success | Check field-level F1 |
| `valid_json_non_schema` | Valid JSON but wrong schema | Inspect model output vs expected |
| `json_parse_error` | Model output wasn't valid JSON | Check `raw` field |
| `degenerate_repetition` | Model stuck in repetition loop | Prompt may be too long or ambiguous |
| `whitespace_run` | Model returned mostly whitespace | Likely a model/endpoint issue |

### Check what the API actually returned

When `valid_json_non_schema` or `all_invalid_responses`, look at the `raw` field to see what the model produced vs what was expected.

---

## Custom Data Sources

### Parquet / CSV / custom format

For non-standard data formats, implement `_load_samples()` in your handler. The handler receives samples in its normalized form. See `src/tasks/document_extraction/traslados_surus.py` for a parquet-based example.

### PDF documents

If the API requires text input, extract PDF text before calling the API:

```python
import subprocess
def _pdf_to_text(self, pdf_path):
    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"],
        capture_output=True, check=False
    )
    return result.stdout.decode("utf-8", errors="replace")
```

### Endpoint selection

If the API returns wrong fields (e.g., invoice fields instead of traslado fields), **switch to the correct endpoint**. The `/extract` endpoint does text+schema; `/factura` does image+schema and returns a fixed invoice schema.

### image_max_edge for multimodal endpoints

Always set `image_max_edge` in the provider config when the API has a pixel-size limit:

```yaml
surus_extract_pdf:
  image_max_edge: 2048   # API hard limit is 2560px
```

Without it, real document images (3000–5000px) get HTTP 400 and manifest as `all_invalid_responses`.

---

## Config Architecture (How Defaults Flow)

Understanding how config reaches the handler is critical for custom datasets:

```
configs/systems/<name>.yaml
  → task_defaults: {}           # top-level defaults for all tasks
  → tasks: ["my_group"]         # which task groups to run

src/tasks/<group>/metadata.yaml
  → subtasks:
      my_subtask:
        source_dir: /path       # MUST be here for custom datasets
        defaults: {}            # per-subtask overrides

build_handler_task_config()      # merges defaults
  → context.defaults           # for BenchmarkRunner (batch_size, etc.)
  → context.subtask_config      # for handler (source_dir, parquet_split, etc.)

Handler.__init__(config)
  → config["source_dir"]        # must be at top level of config dict
```

**Common mistake:** Putting `source_dir` in `configs/systems/<name>.yaml` under a `task_configs` key. The `task_configs` key in system configs is ignored by `build_handler_task_config()`. Always put `source_dir` in `metadata.yaml`'s `subtasks` block.

```yaml
# ✅ Correct — in metadata.yaml
subtasks:
  my_subtask:
    source_dir: /path/to/data

# ❌ Wrong — in system config top-level task_configs
task_configs:
  my_subtask:
    source_dir: /path/to/data   # IGNORED
```

---

## Adding New Providers or Tasks

If your evaluation needs a new model provider or benchmark task, use the dedicated skills first:

- **`add-provider` skill** — for new API endpoints, custom interfaces, provider type wiring
- **`add-task` skill** — for new task handlers, custom data formats, handler implementation

These cover the full code scaffolding so you can focus on the evaluation logic.
