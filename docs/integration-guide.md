# Benchy Integration Guide

This guide is for an agent integrating benchy into a UI app. It tells you what benchy does, how to call it, and what files to read at each stage.

---

## How benchy works (3 sentences)

Benchy evaluates AI systems against a benchmark spec written in YAML. The spec has three stages: define the task, define the scoring, configure the data and model. Once the spec is complete, two CLI commands run the evaluation and produce structured JSON results.

---

## The integration pattern

Your app has editable files and an agent that writes to them. The pattern is:

```
agent edits benchmark.yaml  →  shells out to benchy CLI  →  reads JSON results
```

No Python imports needed. The CLI is the execution layer. The YAML file is the handoff point.

---

## Stages and what the agent writes at each one

Each stage adds a section to `benchmark.yaml`. The file lives at `benchmarks/<name>.yaml`.

### Stage 1 — Define the task
The agent writes the `task:` section: what the AI does, what the input looks like, what the output fields are.

Skill file to read: `.agents/skills/define-task/SKILL.md`

### Stage 2 — Exam design (scoring)
The agent writes the `scoring:` section: how a correct output is measured (per-field, pass/fail, or fuzzy match).

Skill file to read: `.agents/skills/define-scoring/SKILL.md`

### Stage 3 — Data and model configuration
The agent writes the `data:` and `target:` sections: which data to use (or generate), which AI system to evaluate.

Skill files to read:
- `.agents/skills/configure-model/SKILL.md` — for the target (API endpoint, cloud model, or local server)
- `.agents/skills/setup-data/SKILL.md` — for existing data
- `.agents/skills/synthesize-data/SKILL.md` — to generate synthetic examples from the spec

---

## CLI commands (shell out to these)

```bash
# Validate the spec before running
benchy validate --benchmark benchmarks/<name>.yaml

# Smoke test (5 examples, fast)
benchy eval --benchmark benchmarks/<name>.yaml --limit 5 --exit-policy smoke

# Full run
benchy eval --benchmark benchmarks/<name>.yaml --exit-policy strict

# List all benchmark specs in the project
benchy benchmarks
```

Skill file for running: `.agents/skills/run-benchmark/SKILL.md`

---

## Where results land

```
outputs/benchmark_outputs/<run_id>/<benchmark_name>/
├── run_outcome.json      ← overall status: passed / degraded / failed
├── run_summary.json      ← per-field scores and overall score
└── <task>/<subtask>/
    ├── *_metrics.json    ← numeric scores per field
    └── *_samples.json    ← individual predictions with expected vs actual
```

Read `run_outcome.json` first. If status is `passed`, read `run_summary.json` for scores.

Skill file for reading results: `.agents/skills/read-results/SKILL.md`

---

## Environment variables required

| Variable | Used for |
|---|---|
| `TOGETHER_API_KEY` | Data generation (always required); Together AI model targets |
| `OPENAI_API_KEY` | OpenAI model targets |
| `ANTHROPIC_API_KEY` | Anthropic model targets |
| `GOOGLE_API_KEY` | Google model targets |

---

## A complete benchmark.yaml (copy this as a starting point)

```yaml
benchmark:
  name: my-benchmark
  description: One sentence describing what this evaluates

  task:
    type: extraction          # extraction | classification | qa | translation | freeform
    input:
      type: text              # text | image | document
      description: "What the input looks like in plain English"
    output:
      type: structured        # structured | label | text
      fields:
        - name: field_name
          type: string        # string | number | integer | boolean
          description: What this field contains
          required: true

  scoring:
    type: per_field           # per_field | binary | semantic
    case_sensitive: false
    numeric_tolerance: 0.01   # optional, for number fields

  data:
    source: generate          # local | generate | huggingface
    count: 30                 # for generate
    seed_description: "Describe the kind of examples to generate"

  target:
    type: model               # model | api | local
    provider: together        # together | openai | anthropic | google
    model: google/gemma-4-31B-it
    system_prompt: "Optional system prompt"
```

---

## Validate before running

Always run `benchy validate` after writing the spec and before running eval. It returns human-readable errors if anything is missing or wrong. The agent should fix errors using the appropriate stage skill before proceeding.

Skill file: `.agents/skills/validate/SKILL.md`

---

## Reference files

| File | Purpose |
|---|---|
| `coREADME.md` | Full world model: compiler role, mapping tables, multi-benchmark layout |
| `compiler-walkthrough.md` | Three end-to-end scenarios with real terminal output |
| `benchmarks/request-by-email.yaml` | A real working spec to reference |
