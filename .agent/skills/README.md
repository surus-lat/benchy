# Benchy Agent Skills

These are reusable, agent-facing workflows that codify how to do specific
tasks in benchy. They live at `.agent/skills/<name>/SKILL.md` and are
designed to be loaded one at a time when the agent's user starts a
matching task.

## How to use a skill

If you're a Claude Code (or compatible) agent: invoke via the `Skill` tool
with the skill name. The skill content gets loaded into the conversation
and you follow it.

If you're a human contributor: read the `SKILL.md` file directly. They're
written as practical how-tos with concrete commands and expected output.

If you're an agent on a different runtime: read `SKILL.md` directly and
follow the instructions; they don't depend on Claude-specific tooling
beyond standard shell, Python, and git.

## The skills

### Run and interpret benchmarks (most common workflow)

| Skill | One-liner |
|---|---|
| [run-benchmark](./run-benchmark/SKILL.md) | End-to-end: validate spec → smoke → full run → hand off to read-results. The wrapper most users want. |
| [evaluate](./evaluate/SKILL.md) | Execute `benchy eval` against a model or system. Smoke→full workflow, config selection, exit policies. |
| [push-to-latamboard](./push-to-latamboard/SKILL.md) | Merge new benchmark scores into the HuggingFace dataset and make them live on latamboard.surus.lat immediately. |
| [validate](./validate/SKILL.md) | Pre-flight a benchmark spec before running. Reports errors in plain English and routes to the right fixer skill. |
| [interpret-run](./interpret-run/SKILL.md) | Read `run_outcome.json` + `run_summary.json`. Status vocabulary, failure diagnosis, per-task breakdown. |
| [read-results](./read-results/SKILL.md) | Translate benchmark results into plain English for a non-developer audience. |

### Author a benchmark spec (second-layer "describe your AI" workflow)

These four skills compose: a user describes their AI system in plain
English, and the agent walks through each `benchmark.yaml` section in
turn.

| Skill | Spec section it owns |
|---|---|
| [define-task](./define-task/SKILL.md) | `task:` — what the AI does, written from the user's description. |
| [define-scoring](./define-scoring/SKILL.md) | `scoring:` — how to grade the output (exact match, schema, similarity, …). |
| [configure-model](./configure-model/SKILL.md) | `target:` — which AI system to evaluate. |
| [setup-data](./setup-data/SKILL.md) | `data:` — supply test data, with format adaptation. |
| [synthesize-data](./synthesize-data/SKILL.md) | Generate synthetic examples when no real data exists. |

### Extend benchy itself

| Skill | When to use |
|---|---|
| [add-task](./add-task/SKILL.md) | Add a new benchmark task or task group to the repo (`src/tasks/`). |
| [add-provider](./add-provider/SKILL.md) | Add a new inference provider — OpenAI-compatible endpoint or custom HTTP system. |

### Specialized benchmarks

| Skill | What it benchmarks |
|---|---|
| [transcription-benchmark](./transcription-benchmark/SKILL.md) | **Multi-architecture ASR panel** — Whisper (incl. Surus LATAM fine-tune) + Voxtral + Qwen3-ASR + Canary + whisper-1 cloud, on FLEURS es_419 + pt_br. Picks the right venv per model. Validated reference WERs. |
| [whisper-benchmark](./whisper-benchmark/SKILL.md) | Whisper-family ASR on FLEURS es_419 / pt_br locally on a Mac via `transformers_audio`. Includes the MPS large-v3 wedge fix. |
| [qwen3-asr-howto](./qwen3-asr-howto/SKILL.md) | Run Qwen3-ASR (0.6B / 1.7B + ForcedAligner) locally via the `qwen-asr` PyPI package or via DashScope cloud. Covers the transformers<5.0 / Voxtral conflict. |

### Meta

| Skill | Purpose |
|---|---|
| [oracle-plan](./oracle-plan/SKILL.md) | Bidirectional algorithm to convert between a working implementation and a battle-tested design plan. Use for derived design docs and zero-wrong-turn plan execution. |
| [best-part-is-no-part](./best-part-is-no-part/SKILL.md) | Audit design proposals against the principle that the cheapest, simplest part is the one you don't have — while preserving every capability. Catches gold plating disguised as best practice. |

## Authoring a new skill

When you add a new skill:

1. Create `.agent/skills/<kebab-case-name>/SKILL.md`.
2. Start it with YAML frontmatter:

   ```yaml
   ---
   name: <kebab-case-name>
   description: >
     1-3 sentences. State the *task* the skill handles and *when* to use it.
     End with explicit trigger phrases ("Triggers on: …").
   ---
   ```

3. Write the body as a practical how-to. Show concrete commands, expected
   output, and known failure modes. Skills are read by people in a hurry
   (human or agent) — favor terseness and signal over prose.

4. Add a row to the appropriate table in this README.

5. Commit. Skills only help contributors if they're tracked in git.
