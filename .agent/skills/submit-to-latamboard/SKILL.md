---
name: submit-to-latamboard
description: Run a benchmark with benchy and submit the results to LatamBoard via a GitHub PR. The PR merge triggers a GitHub Action that publishes scores to the HuggingFace dataset and makes them live on latamboard.surus.lat immediately. No HuggingFace token required from the contributor.
---

# Submit to LatamBoard

Run a benchmark and publish your scores to [latamboard.surus.lat](https://latamboard.surus.lat) via a PR.

You do not need a HuggingFace token. Scores go live automatically when your PR is merged.

---

## Prerequisites

- benchy installed and working (`benchy --help`)
- Git configured with your name and email
- A fork of `github.com/surus-lat/benchy`
- Your model has a config in `configs/models/` (see `configure-model` skill if not)

---

## Step 1 — Run the benchmark

```bash
benchy eval --config configs/models/<your-model>.yaml --tasks latam_board
```

For transcription models specifically:

```bash
benchy eval --config configs/models/<your-model>.yaml --tasks transcription
```

Note the `run_id` printed at the start of the run (e.g. `20260622_143022_LIMITED`).

Smoke-test first if you want to verify before a full run:

```bash
benchy eval --config configs/models/<your-model>.yaml --tasks latam_board --limit 5 --exit-policy smoke
```

Proceed to full run only if `run_outcome.status` is `passed` or `degraded`.

---

## Step 2 — Process the results

```bash
python -m src.leaderboard.process_all --run-id <run_id>
```

This parses the benchmark outputs and generates `outputs/publish/summaries/<model>_summary.json`.

Check it worked:

```bash
cat outputs/publish/summaries/<model-name>_summary.json | python3 -m json.tool | head -20
```

You should see non-zero scores under `categories`.

---

## Step 3 — Package the submission

```bash
python -m src.leaderboard.package_submission --run-id <run_id> --skip-process
```

This creates:

```
submissions/<run_id>/
  run_manifest.json          ← metadata: model list, date, benchy version
  README.md                  ← exact commands to reproduce
  models/
    <model-name>/
      run_outcome.json       ← proof of execution
      model_summary.json     ← processed leaderboard entry
  configs/
    <model>.yaml             ← the config used (reprex)
```

**Multiple models in one PR:** If you ran multiple models, package each run separately:

```bash
python -m src.leaderboard.package_submission --run-id <run_id_1> --skip-process
python -m src.leaderboard.package_submission --run-id <run_id_2> --skip-process
```

All of them go in the same PR.

---

## Step 4 — Open the PR

```bash
git add submissions/<run_id>/
git commit -m "submission: <run_id> — <model-name>"
git push origin main
```

Then open a PR against `surus-lat/benchy:main`. Use the submission PR template — it includes a checklist for reviewers.

The PR description should include:
- What model(s) you evaluated
- Which task group (`latam_board`, `transcription`, etc.)
- Hardware used (GPU type, VRAM)
- Any degraded tasks and why

---

## Step 5 — After your PR is merged

The GitHub Action fires automatically. It:

1. Loads your `model_summary.json` from `submissions/`
2. Downloads existing scores from HuggingFace
3. Merges your model in (your scores win on conflict)
4. Pushes the updated `leaderboard_table.json` back to HF

Your model appears on [latamboard.surus.lat](https://latamboard.surus.lat) on the next page load — no deploy needed.

---

## Troubleshooting

**`process_all` shows 0.0 scores for all tasks**
The run likely used `_adhoc_structured_*/main/` output layout (modern structured extraction runs). Check that your run's `outputs/benchmark_outputs/<run_id>/<model>/` has the expected task directories.

**`package_submission` says "No summary for <model>"**
Run Step 2 first (`process_all`), then package.

**`package_submission` says "No config found for <model>"**
The model config in `configs/models/` doesn't have a `model.name` matching the full HuggingFace path in `run_outcome.json`. Edit the config to set `model.name: "org/ModelName"` exactly as it appears on HuggingFace.

**PR reviewer asks to reproduce**
They'll run:
```bash
benchy eval --config submissions/<run_id>/configs/<model>.yaml --tasks <task_group> --limit 5
```
Your `submissions/<run_id>/README.md` has the exact commands.

---

## What the reviewer checks

- `run_outcome.json` shows `status: passed` or `degraded` (not `failed`)
- Config in `submissions/<run_id>/configs/` matches what was actually run
- Scores are plausible for the model size and task
- No cherry-picking: if multiple runs exist for the same model on the same date, the PR includes the most recent one
