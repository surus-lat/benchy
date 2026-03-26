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
| `all_invalid_responses` | Wrong model/endpoint or oversized images | Verify model name, endpoint mode; for multimodal endpoints add `image_max_edge` to config if the API has a pixel-size limit (e.g. SURUS /factura requires `image_max_edge: 2048`) |
| exit code 1 + smoke policy | Any task not clean | Read `run_outcome.json` errors section |

---

## Resume a Partial Run

Reuse the same `--run-id` — tasks already marked completed in `task_status.json` are skipped automatically.

```bash
benchy eval --config my-config.yaml --run-id my-run-id --exit-policy strict
```
