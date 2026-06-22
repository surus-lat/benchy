# Workshop Skills Index

Three skills, one per workshop part. Each codifies the exact command sequence so an AI agent can re-run the part for a user on demand.

| Skill | Purpose | Trigger phrases |
|---|---|---|
| [`workshop-define-benchmark`](skills/workshop-define-benchmark/SKILL.md) | Add a new benchy task with custom-weighted EQS scoring (Python handler route) | "define a benchmark", "add a custom extraction task", "score an extraction task my way" |
| [`workshop-benchmark-together-model`](skills/workshop-benchmark-together-model/SKILL.md) | Add a Together AI model and benchmark it on `structured_extraction` | "benchmark a Together model", "add a Together AI model", "run structured_extraction on \<model\>" |
| [`workshop-submit-to-latamboard`](skills/workshop-submit-to-latamboard/SKILL.md) | Package a completed run and open a submission PR (stops at PR; publish handled by organizer) | "submit to latamboard", "publish my benchy results", "open a submission PR" |

## Relationship to upstream skills

| Upstream skill | This workshop skill | Notes |
|---|---|---|
| `.agent/skills/add-task` | `workshop-define-benchmark` | Workshop skill is a concrete instance of the add-task pattern with the paraloq dataset and a reweighted EQS |
| `.agent/skills/{run-benchmark,evaluate}` | `workshop-benchmark-together-model` | Workshop skill inlines the smoke→full pattern using `--tasks structured_extraction` |
| `.agent/skills/submit-to-latamboard` | `workshop-submit-to-latamboard` | Upstream skill documents the full publish flow (currently transcription-only); workshop skill is a one-shot PR-focused sequence that uses `--skip-process` until the extraction processors are modernized |

## Aspirational upstream skills (not currently functional)

These exist in `.agent/skills/` but assume CLI surface that isn't built yet:

| Skill | Why it doesn't work today |
|---|---|
| `define-task`, `define-scoring`, `setup-data`, `configure-model` | They write a `benchmark.yaml` spec consumed by `benchy validate` and `benchy eval --benchmark <path>` — neither exists in the installed CLI |
| `validate` | The `benchy validate` subcommand isn't implemented |

The workshop sidesteps these by using the production `add-task` (Python handler) flow for Part 1.

## Codification

These SKILL.md files are the static canonical version. To register them as live skills in gbrain, an AI agent invokes **gbrain's `/skillify`** at the end of a successful walkthrough — gbrain captures the trigger description, the parameterized command sequence, and recovery hints into the user's skillpack.
