---
name: workshop-submit-to-latamboard
description: Package a completed benchy run as a submission folder and open a PR against surus-lat/benchy. Stops at the PR — actual leaderboard publish is handled separately by the workshop organizer because the extraction processors in src/leaderboard aren't fully modernized yet. Use during Part 3 of the benchy benchmarks workshop.
---

# Workshop — Open a submission PR

This skill walks a user through **Part 3** of the benchy benchmarks workshop: packaging a Part 2 run into a `submissions/<run_id>/` folder, committing it on a feature branch in the user's fork, and opening a PR against `surus-lat/benchy:main`.

**Scope:** this skill stops at the PR. The leaderboard publish (HuggingFace upload + latamboard.surus.lat refresh) is handled by the workshop organizer because the `parse_model_results.py` processors for extraction tasks haven't been updated to consume the current benchy output format. The PR captures the evidence; the organizer finishes the pipeline later.

Companion docs: [`.workshop/parts/part3-submit-to-latamboard.md`](../../parts/part3-submit-to-latamboard.md). Verbose human reference: [`.agent/skills/submit-to-latamboard/SKILL.md`](../../../.agent/skills/submit-to-latamboard/SKILL.md).

## When to invoke

Trigger phrases:
- "submit to latamboard"
- "publish my benchy results"
- "open a submission PR"
- "Part 3 of the benchy workshop"

## Prerequisites

- A completed Part 2 run: `outputs/benchmark_outputs/w2_full_<slot>/<model>/run_outcome.json` exists
- `origin` git remote pointing at the user's fork of `surus-lat/benchy`
- `gh` CLI authenticated (`gh auth status`)
- **No HF token needed** — the publish step is out of scope

## Steps

1. **Pick the run to submit:**
   ```bash
   RUN_ID=w2_full_modelA
   ```

2. **Package the run** with `--skip-process` (the extraction processors aren't ready yet, so skip them):
   ```bash
   .venv/bin/python -m src.leaderboard.package_submission \
     --run-id ${RUN_ID} --skip-process
   ```

3. **Inspect the submission:**
   ```bash
   ls submissions/${RUN_ID}/
   cat submissions/${RUN_ID}/run_manifest.json
   cat submissions/${RUN_ID}/models/*/run_outcome.json | jq '{status, duration_s, model}'
   ```
   Expected files: `run_manifest.json`, `README.md`, `models/<model>/run_outcome.json`, `configs/<model>.yaml`.

4. **Verify `configs/<model>.yaml` is the Together config**, not a vLLM duplicate with the same model name:
   ```bash
   grep -A2 "^together:\|^vllm:" submissions/${RUN_ID}/configs/*.yaml
   ```
   If it picked the wrong one, replace it manually:
   ```bash
   cp configs/models/together_<slot>.yaml submissions/${RUN_ID}/configs/
   ```

5. **Commit on a feature branch and push to the fork:**
   ```bash
   git checkout -b submission/${RUN_ID}
   git add submissions/${RUN_ID}/
   git commit -m "submission: ${RUN_ID} — workshop attendee <name>"
   git push -u origin submission/${RUN_ID}
   ```

6. **Open the PR against `surus-lat/benchy:main`:**
   ```bash
   gh pr create --base main --repo surus-lat/benchy \
     --title "submission: ${RUN_ID} — workshop attendee <name>" \
     --body "Workshop submission. Task: structured_extraction. Model: <model-name>. Hardware: Together AI."
   ```
   Capture the PR URL.

## Captured outputs

Report back:
- Path to `submissions/${RUN_ID}/`
- The PR URL
- The fact that the live publish is pending organizer action (don't claim leaderboard update)

## Recovery hints

- **`package_submission` picks a vLLM config with the same model name:** the script grabs any config whose `model.name` matches. Manually replace with the Together config in step 4.
- **`No summary for <model>`:** expected with `--skip-process`. The submission is valid without a `model_summary.json` — Part 3's scope is the PR, not the publish.
- **`gh pr create` opens against the fork:** always pass `--base main --repo surus-lat/benchy` explicitly.
- **Branch already pushed:** pick a new branch suffix or work on a fresh branch from `main`.
- **PR fails CI:** the publish-submission workflow may fail because `merge_and_publish --from-submissions` can't find an extraction `model_summary.json`. That's expected; the organizer addresses it when modernizing the processors.

## Why the publish is out of scope (instructor cue)

The `process_all` script invokes per-task processors in [`src/leaderboard/functions/parse_model_results.py`](../../../src/leaderboard/functions/parse_model_results.py) that look for legacy `results_*.json` files. Current benchy writes `<model>_<ts>_metrics.json` per subtask instead. Only the transcription processor was updated for the new format (commit `8ffbdcd`). Extraction processors will need the same modernization before the workshop PRs can publish end-to-end. Until then, the workshop ships at "open the PR" and the organizer carries the publish.
