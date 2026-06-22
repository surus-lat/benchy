---
name: workshop-benchmark-together-model
description: Add a new Together AI model to benchy and benchmark it on structured_extraction — smoke first (~30-90s for iteration), then full run for the Part 3 submission. Use when a user wants to benchmark a Together AI model on extraction tasks, especially during the benchy benchmarks workshop.
---

# Workshop — Benchmark a new Together AI model

This skill walks a user through **Part 2** of the benchy benchmarks workshop: adding Together AI models to benchy (zero code — just a YAML) and running them on `structured_extraction` with the smoke→full pattern.

The workshop runs `--tasks structured_extraction` (not `latam_board`) for two reasons: (1) `latam_board`'s `translation` subtask requires gated HF access, (2) the leaderboard processor for spanish/portuguese/structured hasn't been modernized to read the new per-subtask metrics format. Structured-only sidesteps both.

Companion docs: [`.workshop/parts/part2-benchmark-together.md`](../../parts/part2-benchmark-together.md).

## When to invoke

Trigger phrases:
- "benchmark a Together model"
- "add a Together AI model"
- "run structured_extraction on \<model\>"
- "test \<model_id\> on benchy"
- "Part 2 of the benchy workshop"

## Prerequisites

- Working in the benchy repo root
- `TOGETHER_API_KEY` in `.env`
- `jq` and `curl` installed
- benchy installed at `.venv/bin/benchy`

## Steps

1. **Confirm credentials and model availability:**
   ```bash
   echo $TOGETHER_API_KEY | head -c 8 && echo "..."
   for id in google/gemma-3n-E4B-it zai-org/GLM-5.2 openai/gpt-oss-20b; do
     echo -n "$id: "
     curl -s https://api.together.xyz/v1/models \
       -H "Authorization: Bearer $TOGETHER_API_KEY" \
       | jq -r --arg id "$id" '.[] | select(.id==$id) | "✓ \(.type) ctx=\(.context_length)"'
   done
   ```

2. **Install the workshop model configs** (idempotent):
   ```bash
   for slot in modelA modelB modelC; do
     cp .workshop/assets/together_${slot}.yaml.template \
        configs/models/together_${slot}.yaml
   done
   ```

3. **ITERATION LOOP** — smoke each model on `structured_extraction` (~30-90s):
   ```bash
   for slot in modelA modelB modelC; do
     .venv/bin/benchy eval \
       --config configs/models/together_${slot}.yaml \
       --tasks structured_extraction --limit 5 \
       --run-id w2_smoke_${slot} --exit-policy smoke
   done
   ```

4. **Verify smoke statuses:**
   ```bash
   for slot in modelA modelB modelC; do
     echo "=== $slot ==="
     cat outputs/benchmark_outputs/w2_smoke_${slot}_LIMITED/*/run_outcome.json \
       | jq '{status, duration_s, passed_subtasks: .counts.passed_subtasks}'
   done
   ```
   Acceptable: `"passed"` or `"degraded"`. Investigate `"failed"` / `"error"` before proceeding.

5. **SUBMISSION RUN** — full structured_extraction without `--limit` (~2-5 min per model):
   ```bash
   for slot in modelA modelB modelC; do
     .venv/bin/benchy eval \
       --config configs/models/together_${slot}.yaml \
       --tasks structured_extraction \
       --run-id w2_full_${slot} --exit-policy smoke
   done
   ```

6. **Confirm submission runs:**
   ```bash
   for slot in modelA modelB modelC; do
     echo "=== $slot ==="
     cat outputs/benchmark_outputs/w2_full_${slot}/*/run_outcome.json \
       | jq '{status, duration_s, subtasks: .tasks.structured_extraction.subtasks | to_entries | map({name: .key, status: .value.status})}'
   done
   ```

## Captured outputs

Report back:
- Three `configs/models/together_model*.yaml` paths
- Three smoke `run_id`s with `status` and `duration_s`
- Three full `run_id`s with `status` and per-subtask outcomes
- Note any degraded subtasks (chat_extract on GLM-5.2 is expected)

## Recovery hints

- **`benchy: command not found`:** use `.venv/bin/benchy` directly, or activate the venv.
- **401 from Together:** `TOGETHER_API_KEY` missing/invalid. Step 1's curl catches this.
- **404 on a model ID:** Together rotates models. Edit the slot config → swap `model.name` to a current ID → re-smoke.
- **Smoke = `degraded`:** read `run_outcome.json` → `tasks.structured_extraction.subtasks` to see which subtask had partial errors. Typically invalid-JSON responses on small models. Acceptable.
- **Run already exists:** pick a new `--run-id` or `rm -rf outputs/benchmark_outputs/<old_run_id>`.
- **Long stall mid-run:** Together rate-limits sometimes stall multi-subtask runs. Lower `max_concurrent` in `configs/providers/together.yaml`, or wait.

## Why structured_extraction beats latam_board for this workshop (instructor cue)

- `latam_board` includes `translation` which fetches the gated `openlanguagedata/flores_plus` dataset → 403 unless the workshop HF account has access
- The leaderboard processor at `src/leaderboard/functions/parse_model_results.py` only consumes the new per-subtask metrics format for transcription (commit `8ffbdcd`); spanish/portuguese/structured still expect legacy `results_*.json` files that benchy no longer writes
- Until the processor is modernized for the extraction stack, the cleanest "open PR with real data" Part 3 flow is structured_extraction only
