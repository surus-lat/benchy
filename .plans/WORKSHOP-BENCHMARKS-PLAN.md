# Workshop Plan — Benchmarks with Benchy

## Context

Three-part practical workshop on benchmarking with benchy:

1. **Define a custom benchmark** — add a new benchy task at `src/tasks/workshop_extraction/` with custom-weighted EQS scoring (Python handler route via `.agent/skills/add-task/`).
2. **Benchmark new Together AI models** — add 3 small models to `configs/models/` and run them on `--tasks structured_extraction` with the smoke→full pattern.
3. **Open a submission PR** — package the run with `package_submission --skip-process`, commit to `submissions/<run_id>/`, open a PR against `surus-lat/benchy:main`. The workshop stops here; organizer handles the leaderboard publish later.

### Scope reductions discovered during rehearsal

- **Part 1 pivoted from wizard-skill / benchmark.yaml route to Python handler route.** The `benchy validate` and `benchy eval --benchmark <path>` flags assumed by the wizard skills don't exist in the installed CLI.
- **Part 2 pivoted from `--tasks latam_board` to `--tasks structured_extraction`.** Latam_board includes the `translation` subtask which fetches the gated HF dataset `openlanguagedata/flores_plus` (403 without explicit access). Sidestepping it also avoids the half-modernized `parse_model_results.py` processors.
- **Part 3 stops at the PR.** `process_all` only produces a populated `model_summary.json` for transcription runs; the spanish/portuguese/structured processors still expect legacy `results_*.json` files. Organizer modernizes them as followup work, then the workshop PRs publish end-to-end.

Goal: attendees leave having (a) authored a benchmark spec, (b) run a real Together model on the production extraction task, and (c) opened a PR whose merge publishes their scores to `latamboard.surus.lat`.

Chosen modality: **text extraction** (paraloq-family). Easier to debug live than vision tasks, fast smoke runs, no GPU surprises. Image extraction (`facturas`) is held as a bonus.

Part 1 ↔ Part 2 link: **same dataset, custom scoring in Part 1**. Attendees redefine grading on the same data their Part 2 model will be evaluated on — concretely shows scoring is a knob, not a fixed law.

**Each part ships a gbrain skill** so attendees can re-run it with their own AI agent after the workshop. Skills are codified via **gbrain's `/skillify`** (not gstack's `/skillify`, which is browser-scrape oriented). Each skill encodes the exact command sequence for its part — see "Skill" subsection in each part below.

---

## Existing pieces to reuse (do NOT rebuild)

| Asset | Path | Role |
|---|---|---|
| Reference text-extraction task | `src/tasks/structured_extraction/paraloq.py` | Source of dataset + schema for Part 1 |
| Scoring primitives | `src/tasks/common/metrics.py`, `src/tasks/common/utils/structured_metrics_calculator.py` | What `per_field` / `binary` / `semantic` map to internally |
| Benchmark spec wizard skills | `.agent/skills/{define-task,define-scoring,setup-data,configure-model}` | Walks attendees through `benchmarks/<name>.yaml` without writing Python |
| Validation + run | `.agent/skills/{validate,run-benchmark,evaluate}` | `benchy validate`, smoke→strict |
| Together provider (zero code) | `configs/providers/together.yaml` + existing examples `configs/models/together_*.yaml` | Drop a YAML, no code |
| Submission pipeline | `.agent/skills/submit-to-latamboard/SKILL.md`, `src/leaderboard/{process_all,package_submission,merge_and_publish}.py`, `.github/workflows/publish-submission.yml` | Full PR → HF flow, already done |

The submission skill `submit-to-latamboard/SKILL.md` already covers steps 1–5 of Part 3. **Read it first; only add to it if a workshop-specific step is missing** (e.g. forking, troubleshooting Together API keys for first-time attendees).

---

## Workshop skill deliverables (3 gbrain skills)

| # | Skill name | Codifies | Used by attendees to |
|---|---|---|---|
| 1 | `workshop-define-benchmark` | Part 1 command sequence | Author a custom benchmark.yaml for a structured extraction task |
| 2 | `workshop-benchmark-together-model` | Part 2 command sequence | Add a new Together AI model and run it against `latam_board` |
| 3 | `workshop-submit-to-latamboard` | Part 3 command sequence | Package the run as a PR submission and publish to the live leaderboard |

**Codification mechanism: gbrain's `/skillify`** (not gstack's `/skillify`, which targets browser-scrape flows). Process:

1. Do a complete, clean walkthrough of each part live (organizer or rehearsal session).
2. At the end of each part, invoke gbrain's `/skillify` to capture the successful sequence as a permanent skill in the attendee's gbrain skillpack.
3. The skill stores: the trigger description, the parameterized command sequence, expected outputs, and recovery hints from anything that went sideways during the rehearsal.
4. Distribute the 3 skills to attendees ahead of the workshop (or skillify live as a demo of gbrain mechanics).

After the workshop, attendees can re-run any part by asking their AI agent to invoke the corresponding skill — no need to remember the exact command syntax.

---

## Pre-workshop prep (organizer)

1. **Lock the 3 Together models for Part 2.** Run against the live Together API to pick small, fast, chat+structured-capable text models:
   ```bash
   curl -s https://api.together.xyz/v1/models \
     -H "Authorization: Bearer $TOGETHER_API_KEY" \
     | jq '.[] | select(.type=="chat") | {id, context_length, pricing}' \
     | jq -s 'sort_by(.pricing.input_per_million_tokens // 999)[:20]'
   ```
   Pick three with input price ≤ a few cents/M tokens and ≥ 32k context. Sanity candidates from the catalog (verify they still exist): `meta-llama/Llama-3.1-8B-Instruct-Turbo`, `Qwen/Qwen2.5-7B-Instruct-Turbo`, `mistralai/Mistral-7B-Instruct-v0.3`. Existing repo configs (`together_gemma-3n-E4B-it.yaml`, `together_arcee-ai_trinity-mini.yaml`) work as YAML templates.

2. **Pre-write the 3 model YAMLs** in `configs/models/together_<model>.yaml`. Use this skeleton (verbatim from existing configs):
   ```yaml
   model:
     name: <together-api-id>
   together:
     provider_config: together
     overrides:
       temperature: 0.0
       max_tokens: 4096
   task_defaults:
     log_samples: true
   tasks:
   - structured_extraction
   metadata:
     max_context_length: <ctx>
     supports_multimodal: false
     description: "<model> via Together AI"
   ```

3. **Smoke each model in advance** to catch deprecations / 404s before the room sees it:
   ```bash
   benchy eval --config configs/models/together_<id>.yaml \
     --tasks structured_extraction --limit 5 \
     --run-id prep_<id> --exit-policy smoke
   cat outputs/benchmark_outputs/prep_<id>/<model>/run_outcome.json | jq .status
   ```

4. **Provision API keys.** Either workshop-shared `TOGETHER_API_KEY` in `.env`, or instructions for attendees to register and BYO. Note: HF_TOKEN is **not** needed for Part 3 (PR flow handles auth via GitHub Action secret).

5. **Attendees show up with:**
   - benchy installed and `benchy --help` working
   - A fork of `github.com/surus-lat/benchy` with their `origin` pointing at it
   - `TOGETHER_API_KEY` in `.env`
   - Git identity configured

6. **One-page handout** with the exact commands for each part (commands below).

---

## Part 1 — Define a custom benchmark (~25 min)

**Output:** `benchmarks/workshop-extraction.yaml` plus a tiny JSONL of 5 examples drawn from `paraloq/json_data_extraction`.

### Command sequence (full part, copy-pasteable)

```bash
# 0. Start from benchy root with TOGETHER_API_KEY in .env

# 1. Stage the dataset slice (pre-fetched into the workshop fork)
cp workshop/assets/paraloq_5.jsonl benchmarks/data/workshop-extraction.jsonl

# 2. Walk the wizard skills to author benchmarks/workshop-extraction.yaml
#    Each skill writes one section of the YAML — run them in order.
#    (In a Claude/AI session, the attendee invokes them as slash commands.)
/define-task           # writes task: section (input/output schema from paraloq)
/define-scoring        # writes scoring: section — pick per_field, partial_credit:false, case_sensitive:true
/setup-data            # writes data: section — points at benchmarks/data/workshop-extraction.jsonl
/configure-model       # writes target: section — points at one pre-written Together config

# 3. Validate the spec
benchy validate --benchmark benchmarks/workshop-extraction.yaml

# 4. Smoke run (5 samples, forgiving exit policy)
benchy eval --benchmark benchmarks/workshop-extraction.yaml \
  --limit 5 --exit-policy smoke --run-id w1_smoke

# 5. Inspect the result
cat outputs/benchmark_outputs/w1_smoke/<model>/run_outcome.json | jq '.status, .tasks'

# 6. (Stretch) Flip scoring to demonstrate sensitivity
#    Edit benchmarks/workshop-extraction.yaml: partial_credit: true
benchy eval --benchmark benchmarks/workshop-extraction.yaml \
  --limit 5 --exit-policy smoke --run-id w1_smoke_partial
# Compare scores across the two run_ids
```

### Skill to produce: `workshop-define-benchmark`

Codified via **gbrain `/skillify`** at the end of a successful walkthrough. Spec:

- **Name:** `workshop-define-benchmark`
- **Description:** "Author a custom benchy benchmark spec for a structured extraction task: writes benchmark.yaml (task, scoring, data, target sections), stages a tiny dataset, validates, and smoke-runs against a chosen model. Use when a user wants to define their own extraction benchmark from scratch."
- **Triggers on:** "define a benchmark", "custom benchmark spec", "author benchmark.yaml", "score an extraction task my way"
- **Encoded sequence:** the 6 steps above, with the wizard-skill sub-invocations and the validate→smoke→inspect verification loop
- **Captured outputs:** path to the final `benchmarks/*.yaml`, the smoke `run_id`, and the `run_outcome.json` summary

### Steps explained

1. `define-task` — attendees describe what they're extracting; skill writes the `task:` section pointing at the paraloq schema (object with the same fields the production task uses, so Part 2 stays comparable).
2. `define-scoring` — pick `per_field` with `partial_credit: false` and `case_sensitive: true`. This is the **custom** part: it diverges from production's fuzzy EQS (which uses partial F1 with Levenshtein/containment). Attendees see strict per-field scoring produces lower scores than EQS on the same data.
3. `setup-data` — provide 5 rows of `paraloq/json_data_extraction` as a local JSONL. Pre-staged in the workshop fork so attendees `cp` rather than fetch live.
4. `configure-model` — point at one of the 3 pre-written Together configs (so Part 1 has a runnable target without waiting for Part 2).
5. Validate + smoke as shown.

### Talking points

- Why `per_field` ≠ EQS: EQS is `0.2·schema_validity + 0.6·field_f1_partial + 0.2·inverted_hallucination`. Per-field strict is `correct_fields / total_fields`. Same data → different rankings → different conclusions about model quality.
- Where these map internally: `per_field` → `StructuredHandler` + field-level F1; `binary` → `ExactMatch`; `semantic` → `FreeformHandler` + `F1Score`. (Mapping table is in `define-scoring/SKILL.md`.)

**Stretch (if time):** Have attendees flip `partial_credit: true` and re-run; compare scores to demonstrate scoring sensitivity. Or swap to `binary` for whole-record correctness.

---

## Part 2 — Benchmark new Together AI models on the production task (~25 min)

**Output:** Three completed runs in `outputs/benchmark_outputs/<run_id>/<model>/` with `run_outcome.status == "passed"`.

### Command sequence (full part, copy-pasteable)

```bash
# 0. Confirm Together credentials and model list
echo $TOGETHER_API_KEY | head -c 8 && echo "..."
curl -s https://api.together.xyz/v1/models \
  -H "Authorization: Bearer $TOGETHER_API_KEY" \
  | jq '.[] | select(.type=="chat") | .id' | head

# 1. Create or copy a model config (pre-written for the workshop)
ls configs/models/together_*.yaml
#   workshop ships: together_<modelA>.yaml, together_<modelB>.yaml, together_<modelC>.yaml

# 2. Smoke run — 5 samples, forgiving exit policy
MODEL=together_<modelA>
benchy eval --config configs/models/${MODEL}.yaml \
  --tasks structured_extraction --limit 5 \
  --run-id w2_smoke_${MODEL} --exit-policy smoke

# 3. Verify smoke status
cat outputs/benchmark_outputs/w2_smoke_${MODEL}/*/run_outcome.json \
  | jq '{status, counts, error_rate: .tasks | to_entries[0].value.error_rate}'
# Proceed only if status is "passed" or "degraded"

# 4. Full run on the production task bundle that Part 3 publishes
benchy eval --config configs/models/${MODEL}.yaml \
  --tasks latam_board \
  --run-id w2_full_${MODEL} --exit-policy strict

# 5. Confirm the full run
cat outputs/benchmark_outputs/w2_full_${MODEL}/*/run_outcome.json | jq .status

# 6. Repeat steps 2–5 for the other two models (or parallelize across groups)
```

### Skill to produce: `workshop-benchmark-together-model`

Codified via **gbrain `/skillify`** after a clean walkthrough. Spec:

- **Name:** `workshop-benchmark-together-model`
- **Description:** "Add a new Together AI model to benchy and run it against the latam_board production task bundle. Writes the model YAML config under configs/models/, smoke-tests on structured_extraction, then runs the full latam_board eval. Use when a user wants to benchmark a Together AI model on the existing Latin board tasks."
- **Triggers on:** "benchmark a Together model", "add a Together AI model", "run latam_board on <model>", "test <model_id> on benchy"
- **Encoded sequence:** the 6 steps above, parameterized on `MODEL` and `<together-api-id>`
- **Captured outputs:** path to the new `configs/models/together_*.yaml`, the smoke and full `run_id`s, and the `run_outcome.json` summaries

### Talking points

- Why Together is "zero code": OpenAI-compatible. Provider is wired in `connection.py:303-306`; new models = new YAML, full stop.
- `--exit-policy smoke` vs `strict`: smoke is forgiving (lets you see partial failures and continue); strict fails the run on any task error. Always smoke first.
- The `latam_board` task group bundles structured_extraction with other LatamBoard categories — running it is what makes the model "comparable on the leaderboard."

**Stretch (if time):** Swap a model name to a non-existent ID to demonstrate the failure mode and how `run_outcome.json` surfaces it.

---

## Part 3 — Push to LatamBoard via PR (~20 min)

**Output:** One PR per attendee against `surus-lat/benchy:main` containing `submissions/<run_id>/`. On merge, scores appear at `latamboard.surus.lat` within ~1 minute (frontend fetches with `cache: 'no-store'`).

### Command sequence (full part, copy-pasteable)

```bash
# 0. Start from the run_id produced in Part 2 (w2_full_<MODEL>)
RUN_ID=w2_full_<MODEL>

# 1. Process raw outputs into per-model summaries
python -m src.leaderboard.process_all --run-id $RUN_ID

# 2. Inspect the generated summary
cat outputs/publish/summaries/<model-name>_summary.json \
  | jq '.categories | to_entries[] | {name: .key, overall: .value.overall_score}'

# 3. Package into a submission folder
python -m src.leaderboard.package_submission --run-id $RUN_ID --skip-process

# 4. Inspect the submission layout
ls submissions/$RUN_ID/
cat submissions/$RUN_ID/run_manifest.json

# 5. Commit on a feature branch and push to your fork
git checkout -b submission/$RUN_ID
git add submissions/$RUN_ID/
git commit -m "submission: $RUN_ID — <model-name>"
git push -u origin submission/$RUN_ID

# 6. Open the PR (gh CLI shown; web UI also fine)
gh pr create --base main --repo surus-lat/benchy \
  --title "submission: $RUN_ID — <model-name>" \
  --body "Workshop submission. Model: <model-name>. Task: latam_board. Hardware: Together AI (cloud)."

# 7. After merge, watch the action and the live leaderboard
gh run watch --repo surus-lat/benchy
open https://latamboard.surus.lat
```

After merge, `.github/workflows/publish-submission.yml` runs `merge_and_publish.py --from-submissions`, which downloads existing `LatamBoard/leaderboard-results`, merges the new `model_summary.json`, and pushes back. Frontend updates on next page load.

### Skill to produce: `workshop-submit-to-latamboard`

Codified via **gbrain `/skillify`** after a clean end-to-end PR that publishes successfully. Spec:

- **Name:** `workshop-submit-to-latamboard`
- **Description:** "Package a completed benchy run as a LatamBoard submission and open a PR against surus-lat/benchy. The PR merge triggers a GitHub Action that publishes scores to the HuggingFace dataset and makes them live on latamboard.surus.lat. No HF token required. Use when a user has a finished benchy run and wants their model on the public Latin board."
- **Triggers on:** "submit to latamboard", "publish my benchy results", "push to the Latin board", "open a submission PR"
- **Encoded sequence:** the 7 steps above, parameterized on `RUN_ID` and `<model-name>`
- **Captured outputs:** the submission folder path, the PR URL, and the live leaderboard URL after refresh
- **Relationship to existing skill:** wraps the existing `submit-to-latamboard/SKILL.md` (which already documents the manual flow) into a single AI-driven sequence — useful when the existing skill is too verbose to invoke step-by-step in a live workshop

### Talking points

- Why PR-based (not direct HF push): no contributor needs `HF_TOKEN`; reviewers sanity-check `run_outcome.json` and `model_summary.json` before publishing; full reproducibility (the `submissions/<run_id>/configs/<model>.yaml` lets anyone re-run the exact eval).
- Show the GitHub Action firing in real time after the first merge — the moment-of-truth that lands the workshop.

**Logistics:** Have a maintainer ready to merge PRs as they come in, or pre-arrange auto-merge for PRs matching `submissions/**`. Without this, the room cannot see the leaderboard update live.

---

## Stretch goals (only if time permits)

- **Bonus modality (image extraction):** Re-run Part 2 with `--tasks image_extraction` against `Qwen/Qwen2-VL-72B-Instruct` (already configured at `configs/models/together_qwen2.5-vl-72b-instruct.yaml`). Demonstrates multimodal flow on `facturas`. **Note:** depends on factura data state — see "Factura data status" below.
- **Brand-new tiny dataset for Part 1:** Have attendees author 5 JSONL rows by hand on a domain of their choice (resumes, invoices, product reviews) and design a schema + scoring from scratch. Maximally pedagogical but consumes ~20 min.

---

## Factura data status (relevant if pivoting to image extraction)

Two factura tasks exist in the repo:

| Task | Source | Local data state |
|---|---|---|
| `image_extraction/facturas` | User-provided `source_dir` (no HF dataset wired) | **Partial.** `.data/` has `schema.json`, `metrics_config.json`, `dataset_info.json`, and 659 invoice images in `.data/cache/` — but **`datos.json` (ground truth) is missing**. Cannot score without it. |
| `document_extraction/facturas_argentinas` | `mauroibz/facturas_argentinas_2` on HuggingFace (auto-downloaded by `CachedDatasetMixin`) | **Not yet cached locally** (no `facturas_argentinas` subfolder in `.data/`). Will fetch from HF on first run, assuming network + dataset accessible. |

**Implication for the workshop:** If you want to feature image extraction (Part 2 stretch or alternative modality), `facturas_argentinas` is the path of least resistance — first run pulls from HF. The `image_extraction/facturas` task needs you to supply or recover the `datos.json` ground truth before it's runnable. The 659 cached jpgs suggest a prior partial run; check `git log .data/` and any backup of `datos.json` before regenerating it from scratch.

---

## Verification — workshop is deliverable

Run this checklist on the workshop laptop the day before:

- [ ] `benchy --help` works
- [ ] `TOGETHER_API_KEY` set in `.env`; `curl https://api.together.xyz/v1/models -H "Authorization: Bearer $TOGETHER_API_KEY" | head` returns model list
- [ ] All 3 pre-written `configs/models/together_<id>.yaml` files pass `benchy validate` and a 5-sample smoke on `structured_extraction` with `status: passed`
- [ ] `benchy validate --benchmark benchmarks/workshop-extraction.yaml` passes for the Part 1 spec
- [ ] `python -m src.leaderboard.package_submission --run-id <prep_id>` produces a `submissions/<prep_id>/` with `model_summary.json` containing non-zero `categories.<group>.overall_score`
- [ ] A throwaway PR with a prep submission merges cleanly and the workflow `publish-submission.yml` completes green
- [ ] `latamboard.surus.lat` shows the throwaway model after merge (then revert it before the workshop)
- [ ] All 3 gbrain skills (`workshop-define-benchmark`, `workshop-benchmark-together-model`, `workshop-submit-to-latamboard`) exist in the rehearsal account's gbrain skillpack and execute end-to-end when invoked by an AI agent
- [ ] (If using image-extraction stretch) `facturas_argentinas` first-run smoke completes from HF; OR `datos.json` recovered for `image_extraction/facturas`

---

## Logistics decisions (locked)

- **Submission flow:** each attendee forks `surus-lat/benchy` and opens their own PR. Drops the "shared workshop branch" shortcut — the real PR experience is part of the lesson.
- **Issue tracking:** no Plane integration. PRs are the only paper trail.
- **Model picks:** organizer to confirm the 3 Together model IDs against the live catalog before shipping the workshop materials.

---

## Distribution: `.workshop/` folder (post-test)

Once the workshop has been **fully rehearsed end-to-end and all 3 parts verified**, package the materials into a `.workshop/` directory at the repo root and commit. Goal: any user can later say "walk me through the benchy workshop" to their AI agent and it has everything needed.

### Layout

```
.workshop/
  README.md                          ← entry point: what the workshop is, prerequisites, suggested order
  SKILLS-INDEX.md                    ← one-paragraph summary per skill with the trigger phrases
  parts/
    part1-define-benchmark.md        ← Part 1 walkthrough + command sequence (mirrors plan)
    part2-benchmark-together.md      ← Part 2 walkthrough + command sequence
    part3-submit-to-latamboard.md    ← Part 3 walkthrough + command sequence
  skills/
    workshop-define-benchmark/SKILL.md
    workshop-benchmark-together-model/SKILL.md
    workshop-submit-to-latamboard/SKILL.md
  assets/
    paraloq_5.jsonl                  ← pre-staged dataset slice for Part 1
    together_<modelA>.yaml           ← copies of the 3 model configs (also installed at configs/models/)
    together_<modelB>.yaml
    together_<modelC>.yaml
    workshop-extraction.example.yaml ← reference benchmark spec attendees produce in Part 1
  rehearsal-notes.md                 ← what went sideways during testing + how it was recovered (gold for future skillification)
```

### Conventions

- Skill SKILL.md files use the same frontmatter format as `.agent/skills/<name>/SKILL.md` so they can be invoked directly by AI agents that read either location.
- `assets/together_<model>.yaml` files are **copies** of the production configs in `configs/models/` — keep them in sync if the model IDs change.
- `README.md` should explicitly state: "this folder is the source of truth for the workshop; the plan in `.plans/` is the planning record."

### Trigger to ship

The "Verification — workshop is deliverable" checklist (above) must be all-green on the rehearsal laptop. Only then create `.workshop/`, commit, and push to `main`. The commit message: `docs: add .workshop/ — benchy benchmarks workshop materials`.
