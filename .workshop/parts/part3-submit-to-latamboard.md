# Part 3 — Open a submission PR

**Time:** ~20 min  
**Output:** A PR against `surus-lat/benchy:main` containing `submissions/<run_id>/`  
**Skill:** [`skills/workshop-submit-to-latamboard/SKILL.md`](../skills/workshop-submit-to-latamboard/SKILL.md)

---

## Scope (read carefully)

This part stops at **opening the PR**. The PR contains:
- `run_outcome.json` from Part 2's full run (proof of execution)
- The model config used (reproducibility)
- A manifest with the run ID and model list

The workshop organizer handles the actual leaderboard publish later. Why: `process_all` and `merge_and_publish` expect a `model_summary.json` per model that the leaderboard pipeline can't currently generate for extraction tasks (only the transcription processor was modernized). The publish becomes follow-up engineering work; the PR is the artifact for that work.

---

## Step 1 — Pick the run to submit

You probably have three: `w2_full_modelA`, `w2_full_modelB`, `w2_full_modelC`. You can submit one or all three. We'll show the flow for one; loop the same commands for the others.

```bash
RUN_ID=w2_full_modelA
```

---

## Step 2 — Package the run

```bash
.venv/bin/python -m src.leaderboard.package_submission \
  --run-id ${RUN_ID} --skip-process
```

> **Why `--skip-process`:** without it, `package_submission` first runs `process_all`, which currently emits warnings like `No results_*.json files found in <task_dir>` because the extraction processors haven't been updated. With `--skip-process`, we skip straight to packaging the raw evidence.

Expected output ends with:
```
✅ Submission packaged → submissions/w2_full_modelA
   1 model(s): <model-name>

Next steps:
  git add submissions/w2_full_modelA
  git commit -m 'submission: w2_full_modelA (1 models)'
  git push && open a PR
```

---

## Step 3 — Inspect what got packaged

```bash
ls submissions/${RUN_ID}/
```

Expected structure:
```
submissions/w2_full_modelA/
├── README.md                              # exact reproduction commands
├── run_manifest.json                      # metadata: run_id, models, packaged_at
├── configs/
│   └── <model-config>.yaml                # the config you used
└── models/
    └── <model-name>/
        └── run_outcome.json               # proof of execution + per-task statuses
```

Verify the manifest:
```bash
cat submissions/${RUN_ID}/run_manifest.json
```

Expected:
```json
{
  "run_id": "w2_full_modelA",
  "packaged_at": "2026-XX-XXTXX:XX:XXZ",
  "models": ["google/gemma-3n-E4B-it"],
  "model_count": 1
}
```

Verify the run outcome:
```bash
cat submissions/${RUN_ID}/models/*/run_outcome.json \
  | jq '{status, duration_s, model, subtasks: .tasks.structured_extraction.subtasks | to_entries | map({name: .key, status: .value.status})}'
```

---

## Step 4 — Sanity-check the packaged config

`package_submission` grabs *any* config in `configs/models/` whose `model.name` matches your run's model. If you have a vLLM config and a Together config for the same model name, it may pick the wrong one. Check:

```bash
grep -E "^together:|^vllm:|^openai:" submissions/${RUN_ID}/configs/*.yaml
```

You want to see `together:`. If you see `vllm:` instead, fix it manually:

```bash
cp configs/models/together_modelA.yaml submissions/${RUN_ID}/configs/
# (and `rm` the wrong one if needed)
```

---

## Step 5 — Commit on a feature branch

Make sure your `origin` points at your fork, not upstream:

```bash
git remote -v
# should show: origin <your-fork-url>  (fetch)
#              origin <your-fork-url>  (push)
```

If it shows `surus-lat/benchy.git` for origin, fix it first:
```bash
git remote set-url origin git@github.com:<your-github-user>/benchy.git
```

Then commit and push:

```bash
git checkout -b submission/${RUN_ID}
git add submissions/${RUN_ID}/
git commit -m "submission: ${RUN_ID} — <your-name>"
git push -u origin submission/${RUN_ID}
```

---

## Step 6 — Open the PR against `surus-lat/benchy:main`

Using the `gh` CLI (fast):

```bash
gh pr create --base main --repo surus-lat/benchy \
  --title "submission: ${RUN_ID} — <your-name>" \
  --body "$(cat <<'EOF'
Workshop submission.

- Task: `structured_extraction` (3 subtasks: paraloq, chat_extract, email_extract)
- Model: <model-name>
- Provider: Together AI
- Hardware: cloud (Together-hosted)
- Workshop date: <YYYY-MM-DD>

Generated via the benchy benchmarks workshop. `model_summary.json` is intentionally absent — extraction leaderboard processors are pending modernization. The maintainer will finish the publish once that lands.
EOF
)"
```

Or via the web UI: push the branch (Step 5), then visit `https://github.com/<your-github-user>/benchy/compare/main...submission/${RUN_ID}` and click **Create pull request** → change base to `surus-lat/benchy:main` → fill in title and body using the same wording.

Capture the PR URL. **That URL is the workshop deliverable.**

---

## What "done" looks like

- `submissions/<run_id>/` exists with `run_manifest.json`, `README.md`, `models/<model>/run_outcome.json`, `configs/<model>.yaml`
- The packaged `configs/<model>.yaml` is the Together config (not a vLLM duplicate)
- A PR is open against `surus-lat/benchy:main` from your fork
- You have the PR URL

## Common stumbles

- **`No summary for <model>` warning:** expected. `--skip-process` suppresses the summary-generation step that doesn't yet work for extraction.
- **`gh pr create` opens against your fork instead of upstream:** you forgot `--base main --repo surus-lat/benchy`. Always specify both.
- **`Updates were rejected because the remote contains work...`:** your branch is out of date with your fork. `git fetch origin && git rebase origin/main` and try again.
- **Branch already pushed (re-doing this part):** pick a new branch suffix like `submission/${RUN_ID}-v2`, or delete the old one with `git push origin --delete submission/${RUN_ID}` first.
- **`package_submission` grabbed the wrong config:** see Step 4. The script matches on `model.name`; manually replace if you have duplicate configs for the same model.

## What happens after merge (organizer-side, not in workshop)

1. The `publish-submission.yml` GitHub Action fires on push to main when files in `submissions/**` change.
2. The action runs `merge_and_publish.py --from-submissions` which expects `model_summary.json` files per model.
3. **Today:** the action will complete with the structured/spanish/portuguese submissions producing empty summaries; only transcription submissions publish. The organizer either patches `parse_model_results.py` to read the new format, or runs a manual processing step against the raw `run_outcome.json` evidence.
4. **Future:** once the processors are modernized (mirror the transcription processor at commit `8ffbdcd`), workshop PRs publish end-to-end without manual intervention.
