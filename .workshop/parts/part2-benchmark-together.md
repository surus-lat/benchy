# Part 2 — Benchmark new Together AI models on extraction

**Time:** ~25 min  
**Output:** Three smoke runs (~30-90s each) + one full `structured_extraction` run per model  
**Skill:** [`skills/workshop-benchmark-together-model/SKILL.md`](../skills/workshop-benchmark-together-model/SKILL.md)

---

## Goal

Add three small Together AI models to benchy and benchmark them on `structured_extraction` (paraloq + chat_extract + email_extract). The lesson: Together is OpenAI-compatible, so adding a model is **zero code** — just a YAML file.

## Why structured_extraction and not latam_board?

The wider `latam_board` task group is currently blocked for the workshop:
- The `translation` subtask pulls the gated HF dataset `openlanguagedata/flores_plus` (403 unless your HF account has access)
- The leaderboard processor for spanish/portuguese/structured hasn't been modernized for the new per-subtask metrics format

`--tasks structured_extraction` sidesteps both issues.

## Model picks (locked, in `.workshop/assets/`)

| Slot | Model | Rehearsal smoke (5 samples) |
|---|---|---|
| A | `google/gemma-3n-E4B-it` | 41s, all 3 subtasks passed |
| B | `zai-org/GLM-5.2` | 73s, paraloq+email passed, chat_extract degraded (3/5 valid) |
| C | `openai/gpt-oss-20b` | 51s, all 3 subtasks passed |

---

## Step 1 — Verify credentials and that the model IDs resolve on Together

```bash
echo $TOGETHER_API_KEY | head -c 8 && echo "..."

for id in google/gemma-3n-E4B-it zai-org/GLM-5.2 openai/gpt-oss-20b; do
  echo -n "$id: "
  curl -s https://api.together.xyz/v1/models \
    -H "Authorization: Bearer $TOGETHER_API_KEY" \
    | jq -r --arg id "$id" '.[] | select(.id==$id) | "✓ \(.type) ctx=\(.context_length)"'
done
```

Expected output:
```
google/gemma-3n-E4B-it: ✓ chat ctx=32768
zai-org/GLM-5.2: ✓ chat ctx=128000
openai/gpt-oss-20b: ✓ chat ctx=131072
```

If any returns blank, the model has been rotated/deprecated on Together. Pick a replacement from `curl ... | jq '.[] | select(.type=="chat") | .id'` and update the corresponding template in Step 2.

---

## Step 2 — Install the three workshop model configs

You can either `cp` the templates straight in (fast), or write them by hand to see what each block does. Here's **slot A** in full, so you understand what you're committing to:

**`configs/models/together_modelA.yaml`** (the file you create):

```yaml
model:
  name: google/gemma-3n-E4B-it

together:
  provider_config: together  # references configs/providers/together.yaml
  overrides:
    temperature: 0.0
    max_tokens: 4096

task_defaults:
  log_samples: true

tasks:
- structured_extraction
- latam_board

metadata:
  max_context_length: 32768
  supports_multimodal: true
  supports_schema: true
  provider: together
  model_type: gemma3
  description: "gemma-3n-E4B-it via Together AI — small multimodal model"
```

### What each block means

| Block | Purpose |
|---|---|
| `model.name` | The exact model ID Together's chat completions API expects |
| `together.provider_config: together` | Points at `configs/providers/together.yaml` for base_url, API key env var (`TOGETHER_API_KEY`), retry config |
| `together.overrides` | Per-model overrides for the provider defaults — keep temperature deterministic for benchmarking |
| `task_defaults.log_samples: true` | Save per-sample request/response pairs (you'll want these for debugging) |
| `tasks` | The task groups this model is *intended* for. Doesn't restrict what you can run via `--tasks`. |
| `metadata` | Capabilities the leaderboard uses for filtering/grouping |

### Install all three at once (the shortcut)

```bash
for slot in modelA modelB modelC; do
  cp .workshop/assets/together_${slot}.yaml.template \
     configs/models/together_${slot}.yaml
done

ls configs/models/together_model*.yaml
```

Expected: three files. The B and C templates differ only in `model.name`, `metadata.max_context_length`, `metadata.model_type`, and `metadata.description`. Take a moment to open `configs/models/together_modelB.yaml` and `configs/models/together_modelC.yaml` and confirm what's there.

---

## Step 3 — Smoke each model on structured_extraction (~30-90s each)

```bash
for slot in modelA modelB modelC; do
  echo "=== Smoking $slot ==="
  .venv/bin/benchy eval \
    --config configs/models/together_${slot}.yaml \
    --tasks structured_extraction --limit 5 \
    --run-id w2_smoke_${slot} --exit-policy smoke
done
```

Wall-clock: roughly 30s + 75s + 50s = **~2-3 min total** for all three.

> **Don't run all three at once via `&`.** Together rate-limits aggressive parallel calls and the runs share `.data/` cache writes. Sequential is fine.

---

## Step 4 — Read the smoke results

```bash
for slot in modelA modelB modelC; do
  echo "=== $slot ==="
  cat outputs/benchmark_outputs/w2_smoke_${slot}_LIMITED/*/run_outcome.json \
    | jq '{status, duration_s, passed: .counts.passed_subtasks}'
done
```

Expected output (close to this):
```
=== modelA ===
{ "status": "passed", "duration_s": 41.0, "passed": 3 }
=== modelB ===
{ "status": "degraded", "duration_s": 73.4, "passed": 2 }
=== modelC ===
{ "status": "passed", "duration_s": 51.6, "passed": 3 }
```

**Slot B (GLM-5.2) will come back `degraded`.** That's real — it throws invalid JSON on 2 of 5 chat_extract samples. Look at the breakdown:

```bash
cat outputs/benchmark_outputs/w2_smoke_modelB_LIMITED/*/run_outcome.json \
  | jq '.tasks.structured_extraction.subtasks | to_entries | map({name: .key, status: .value.status, valid: .value.summary.valid_samples})'
```

That degradation **is the signal**. The benchmark surfaced a real model-fit issue in 73 seconds. You'd want to know that before publishing a GLM-5.2 score.

---

## Step 5 — Submission run (no `--limit`, ~2-5 min per model)

```bash
for slot in modelA modelB modelC; do
  echo "=== Full run for $slot ==="
  .venv/bin/benchy eval \
    --config configs/models/together_${slot}.yaml \
    --tasks structured_extraction \
    --run-id w2_full_${slot} --exit-policy smoke
done
```

Wall-clock: **~2-5 min per model** depending on the model and the chat_extract size. Total: ~10-15 min for all three.

---

## Step 6 — Confirm the submission runs

```bash
for slot in modelA modelB modelC; do
  echo "=== $slot ==="
  cat outputs/benchmark_outputs/w2_full_${slot}/*/run_outcome.json \
    | jq '{status, duration_s, subtasks: .tasks.structured_extraction.subtasks | to_entries | map({name: .key, status: .value.status, valid: .value.summary.valid_samples})}'
done
```

Each `w2_full_<slot>` should have a non-empty `subtasks` list. That's what Part 3 packages.

---

## What "done" looks like

- Three `configs/models/together_model*.yaml` files exist
- Three `w2_smoke_<slot>_LIMITED/<model>/run_outcome.json` files with `status: "passed"` or `"degraded"`
- Three `w2_full_<slot>/<model>/run_outcome.json` files from the submission runs
- You can name a subtask that came back degraded and explain why

## Common stumbles

- **`benchy: command not found`:** use `.venv/bin/benchy`. benchy ships in the project venv.
- **401 from Together:** `TOGETHER_API_KEY` missing or wrong. Step 1's curl catches this before you spend money.
- **404 on a model ID:** Together rotates small models without warning. Open the slot YAML, swap `model.name` to a current ID from Step 1, re-smoke.
- **Smoke shows `degraded`:** read `run_outcome.json` → `tasks.structured_extraction.subtasks` for the per-subtask breakdown. Usually invalid-JSON responses on small models. Acceptable to publish.
- **`Run already exists`:** benchy refuses to overwrite. Pick a new `--run-id` or `rm -rf outputs/benchmark_outputs/<old_run_id>`.
- **Long stall mid-run:** Together rate-limits sometimes stall multi-subtask runs. Lower `max_concurrent` in `configs/providers/together.yaml` to 1 or 2, or wait.

## Stretch

Swap a `model.name` to a non-existent ID (e.g. `meta-llama/Llama-9000-Hypothetical`), re-smoke, read `run_outcome.json` to see the failure surface. Good for grounding in benchy's diagnostics.
