---
name: push-to-latamboard
description: Publish new benchmark results from benchy to the LatamBoard leaderboard. Merges new model scores with the existing HuggingFace dataset and makes them live on latamboard.surus.lat immediately — no frontend redeploy needed.
---

# Push to LatamBoard

Publishes new benchmark results to [latamboard.surus.lat](https://latamboard.surus.lat).

---

## How the stack fits together

```
benchy eval                          ← runs the model
  └─ outputs/benchmark_outputs/<run_id>/<model>/

merge_and_publish.py                 ← leaderboard pipeline
  ├─ downloads existing scores from HuggingFace
  ├─ merges new results on top
  └─ pushes merged data back to HuggingFace

HuggingFace dataset                  ← source of truth
  mauroibz/leaderboard-results
  └─ leaderboard_table.json          ← one row per model

latamboard frontend (runtime fetch)  ← live, no redeploy
  └─ fetches leaderboard_table.json from HF on every page load
```

The frontend fetches `leaderboard_table.json` live from HuggingFace with
`cache: 'no-store'`. **Once benchy pushes, the leaderboard updates on the
next browser refresh** — no frontend build or deploy required.

---

## Prerequisites

**One-time setup:** add your HuggingFace token to `.env` in the benchy root:

```bash
echo "HF_TOKEN=hf_..." >> .env
```

The token needs write access to `mauroibz/leaderboard-results`.

Verify the dataset name in `configs/config.yaml` matches what the frontend
pulls:

```yaml
# configs/config.yaml
datasets:
  results: "mauroibz/leaderboard-results"   # must match latamboard's fetch URL
```

> **Note:** benchy was historically configured with `LatamBoard/leaderboard-results`.
> The latamboard frontend actually fetches from `mauroibz/leaderboard-results`.
> If these differ, update `configs/config.yaml` to match the latamboard frontend.

---

## Step 1 — Run the benchmark

If you don't have a completed run yet, use the `run-benchmark` or `evaluate`
skill to produce one. The run ID is printed at the start of every `benchy eval`
run and also appears in `outputs/benchmark_outputs/`.

```bash
benchy eval --config <model-config> --tasks latam_board --run-id <run_id>
```

Confirm the run passed:

```bash
cat outputs/benchmark_outputs/<run_id>/<model_name>/run_outcome.json | python3 -m json.tool | grep '"status"'
# expected: "status": "passed"
```

---

## Step 2 — Merge and publish

```bash
python -m src.leaderboard.merge_and_publish --run-id <run_id>
```

This runs the full pipeline:

1. Processes the run into `outputs/publish/summaries/`
2. Downloads existing scores from HuggingFace
3. Merges: new model results overwrite old entries for the same model key
4. Regenerates `leaderboard_table.json` (all models, all columns)
5. Pushes everything back to HuggingFace

Expected output ends with:

```
🎉 Done! View the dataset at:
   https://huggingface.co/datasets/mauroibz/leaderboard-results
```

If the run was already processed and `outputs/publish/` is up to date:

```bash
python -m src.leaderboard.merge_and_publish --skip-process
```

---

## Step 3 — Verify

Open [latamboard.surus.lat](https://latamboard.surus.lat) and hard-refresh
(`Cmd+Shift+R`). The new model row should appear immediately.

To verify the HuggingFace side directly:

```bash
python3 -c "
import json
from huggingface_hub import hf_hub_download
p = hf_hub_download('mauroibz/leaderboard-results', 'leaderboard_table.json', repo_type='dataset')
data = json.load(open(p))
print([m['model_name'] for m in data])
"
```

---

## What gets updated automatically vs. what needs a frontend redeploy

| File | Fetch method | Auto-updates? |
|------|-------------|---------------|
| `leaderboard_table.json` | Live HF fetch on every page load | ✅ Yes — instant |
| `tasks_groups.json` | Served from `/public/` (build artifact) | ❌ Needs redeploy |
| `tasks_list.json` | Served from `/public/` (build artifact) | ❌ Needs redeploy |

You only need to redeploy the latamboard frontend when **task definitions
change** (new tasks, new task groups, updated descriptions). For score updates
alone, Step 2 is sufficient.

---

## Troubleshooting

**`❌ HF_TOKEN not set`** — Add `HF_TOKEN=hf_...` to `.env` in the benchy root.

**Model shows 0.0 scores** — The run likely used the new `_adhoc_structured_*/main/`
output layout. Confirm with:
```bash
ls outputs/benchmark_outputs/<run_id>/<model_name>/
```
If you see `_adhoc_structured_*/` dirs instead of `structured_extraction/`,
the processor should handle them automatically (fixed in commit `3c8bdd7`).

**Model appears with `publisher: unknown`** — The run has no `run_config.yaml`.
Confirm `run_outcome.json` exists and contains a `"model"` key:
```bash
cat outputs/benchmark_outputs/<run_id>/<model_name>/run_outcome.json | python3 -m json.tool | grep '"model"'
```
If the key is present, the fallback in `extract_model_info_from_config` will
use it (fixed in commit `8740764`).

**Old models disappeared from the leaderboard** — `merge_and_publish` merges
new results on top of the existing HF dataset; it never deletes old entries.
If scores vanished, the HF download step may have failed silently. Check the
output of Step 2 for `❌` lines.

**Dataset name mismatch** — If `configs/config.yaml` has `LatamBoard/leaderboard-results`
but the frontend fetches from `mauroibz/leaderboard-results`, pushes go to the
wrong dataset. Update `configs/config.yaml` to use `mauroibz/leaderboard-results`.
