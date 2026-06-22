# Benchy Benchmarks Workshop

Hands-on materials for the three-part benchy benchmarking workshop. Designed so any AI agent (Claude Code, etc.) can walk a user through each part end-to-end.

**Source of truth:** this folder. The planning record is in [`../.plans/WORKSHOP-BENCHMARKS-PLAN.md`](../.plans/WORKSHOP-BENCHMARKS-PLAN.md).

---

## What attendees build

| # | Part | Output |
|---|---|---|
| 1 | [Define a custom benchmark](parts/part1-define-benchmark.md) | New task at `src/tasks/workshop_extraction/` with custom-weighted EQS scoring |
| 2 | [Benchmark new Together AI models](parts/part2-benchmark-together.md) | Three completed `structured_extraction` runs on small Together models |
| 3 | [Open a submission PR](parts/part3-submit-to-latamboard.md) | A PR per attendee against `surus-lat/benchy:main` containing the run artifacts |

Each part has a matching skill in [`skills/`](skills/) that an AI agent can invoke to walk the user through it. See [`SKILLS-INDEX.md`](SKILLS-INDEX.md).

> **Scope note for Part 3:** the workshop ends at the PR. The live leaderboard publish (HuggingFace upload + latamboard.surus.lat refresh) is out of scope until the `parse_model_results.py` processors for extraction tasks are modernized — see [`parts/part3-submit-to-latamboard.md`](parts/part3-submit-to-latamboard.md) for details and the followup work for the organizer.

---

## Prerequisites (attendees)

- benchy installed (use `.venv/bin/benchy` since it ships in the project venv)
- A fork of `github.com/surus-lat/benchy` with `origin` pointing at it
- `TOGETHER_API_KEY` in `.env`
- Git identity configured + `gh` CLI authenticated

You do **not** need an HF token. Part 3 stops at the PR; the publish is handled later by the organizer.

---

## How to use this with your AI agent

Ask your AI to run the workshop:

> "Walk me through the benchy workshop. Start with Part 1."

The AI should:

1. Read [`parts/part1-define-benchmark.md`](parts/part1-define-benchmark.md) (then 2, then 3).
2. Invoke the matching skill in [`skills/`](skills/) when it gets to each part.
3. Stop after each part for you to confirm before moving on.

If your AI uses gbrain, the skills in [`skills/`](skills/) are in the canonical SKILL.md format and can be loaded directly. Other skill mechanisms (Claude Code's `.agent/skills/`, etc.) use the same format and work as-is.

---

## Workshop assets

- [`assets/together_modelA.yaml.template`](assets/together_modelA.yaml.template), `_modelB`, `_modelC` — the 3 Together model configs (slot A=gemma-3n-E4B-it, slot B=GLM-5.2, slot C=gpt-oss-20b)
- [`assets/workshop_extraction_extract.py.example`](assets/workshop_extraction_extract.py.example) — Python handler for Part 1
- [`assets/workshop_extraction_metadata.yaml.example`](assets/workshop_extraction_metadata.yaml.example) — task metadata for Part 1
- [`rehearsal-notes.md`](rehearsal-notes.md) — what went sideways during rehearsal and what was learned

---

## Maintenance

- When the 3 Together model IDs change, update `.workshop/assets/together_<slot>.yaml.template` (the canonical workshop copies). The production `configs/models/together_<slot>.yaml` files are created at workshop time from these templates.
- When `add-task/SKILL.md` or `submit-to-latamboard/SKILL.md` change upstream, re-sync `skills/workshop-*/SKILL.md`.
- Re-rehearse end-to-end whenever benchy's CLI surface or the leaderboard pipeline changes shape.
- When the leaderboard processors are modernized for extraction (see Part 3's "Future work" section), drop `--skip-process` from Part 3's command sequence.
