# Second Layer: Compiler + Agent Skill Plan

## What This Is

The second layer sits above the engine and exposes benchmarking to non-developer users. It has two parts:

1. **`benchmark.yaml`** — a human-vocabulary spec file that lives at the project root. Four sections: `task`, `scoring`, `target`, `data`.
2. **`benchmark_compiler.py`** — translates `benchmark.yaml` into a handler instance that plugs into the existing `BenchmarkRunner`. Zero engine changes.
3. **7 agent skills** — SKILL.md files that tell the AI agent what to do and what to produce at each stage.

Reference: `docs/SECOND_LAYER_DESIGN.md` has the full spec including `benchmark.yaml` format, compiler mapping rules, and build order.

---

## The 3-Stage Workflow

The user never sees benchy internals. The stages follow how someone thinks about testing an AI system.

```
Stage 1 — Define Benchmark    what does the AI do?              →  task: section
Stage 2 — Exam Planning        how do we grade it?               →  scoring: section
Stage 3 — Run Setup            who takes the exam + the papers   →  target: + data: sections
           └─ configure-model  which AI system
           └─ setup-data       local file | adapt format | synthesize
```

Stage 3 pairs model and data because they belong together: "who's taking the exam" and "what does the exam look like" are the same decision.

`benchmark.yaml` lives at the project root. One benchmark per project.

---

## Skill Inventory (7 skills)

### Stage 1: `define-task`
*(Standalone)*

Collects the task definition. What the AI is supposed to do.

Questions the agent asks:
- What does your AI do? (plain English → maps to task type internally)
- What goes in? (text | image | document)
- What comes out? (specific fields | a category | free text)
- If fields: list them (name, type, description, required)

**Output artifact:** `task:` section of `benchmark.yaml`

Internal mapping (never shown to user):
- extraction + structured output → `structured` task type
- classification + label → `classification` task type
- qa / translation + text → `freeform` task type
- any of the above + image input → `multimodal_input: true` on the config (not a separate handler)

---

### Stage 2: `define-scoring`
*(Standalone)*

Helps the human choose how to grade the AI's output. No math required.

Three options in plain English:
1. **Per field** — "3 of 4 fields correct = score of 0.75." Use for extraction tasks.
2. **Pass/fail** — "All fields correct = pass, otherwise fail." Use when partial credit doesn't make sense.
3. **Fuzzy match** — "Close enough counts." Use for QA, translation, summaries.

Follow-up params (only when needed):
- Per field: tolerate numeric rounding? case-sensitive?
- Pass/fail: case-sensitive?
- Fuzzy: nothing extra

**Output artifact:** `scoring:` section of `benchmark.yaml`

---

### Stage 3a: `configure-model`
*(Standalone. Can be run before or after `setup-data`.)*

Captures which AI system is being evaluated. User vocabulary: "your API", "a model", "a local server." The word "target" never appears.

Three paths:
1. **Your API endpoint** — URL, body template with `{{field}}` placeholders, where to find the answer in the response. `GenericAPIInterface` handles this internally.
2. **A model** — provider (OpenAI, Anthropic, Together, etc.), model name, system prompt.
3. **A local server** — URL of a local OpenAI-compatible endpoint + model name. (Script/subprocess path is out of scope for v1.)

Generator model (used in `synthesize-data`) is an engine detail — never surfaced here or anywhere.

**Output artifact:** `target:` section of `benchmark.yaml` (key name is internal only)

---

### Stage 3b: `setup-data`
*(Standalone. Can be run before or after `configure-model`.)*

Supplies the benchmark examples. Three paths:
1. **Plug and run** — user has a JSONL or CSV. Agent validates required fields (`id`, `text` or `image_path`, `expected`), gives a clear error if something's missing.
2. **Adapt format** — user has data but in the wrong shape. Agent maps their columns to the required fields, writes a clean JSONL.
3. **Synthesize** — no data. Agent invokes the `synthesize-data` skill.

Image task hard rule: if task input is image and user picks synthesize, block it — "I can generate expected outputs but not the images themselves. You need real images — let me help you format them instead."

**Output artifact:** `data:` section of `benchmark.yaml`

---

### Utility: `synthesize-data`
*(Standalone. Invoked from `setup-data` when there's no data, or directly: "generate 30 examples for my invoice benchmark".)*

Generates synthetic `{input, expected}` pairs from the task spec.

Steps:
1. Read `task:` section from `benchmark.yaml` (type, field schema, optional seed description)
2. Build generation prompt
3. Call generator model N times in parallel (asyncio, not sequential)
4. Stream progress to terminal: "Generated 12/30..."
5. Validate each pair against field schema
6. Save to `.data/<benchmark-name>/train.jsonl`

Generator model is an engine detail — not configurable by the user, not mentioned in output.

**Output artifact:** `.data/<benchmark-name>/train.jsonl`

---

### Execution: `run-benchmark`
*(Standalone)*

Runs the benchmark end-to-end from a complete `benchmark.yaml`.

Steps:
1. Verify all 4 sections are present in `benchmark.yaml`
2. Smoke run: `benchy eval --benchmark <name> --limit 5 --exit-policy smoke`
3. Parse `run_outcome.json` — check gates (exit_code 0, status passed/degraded, all failure counts 0)
4. If clean: `benchy eval --benchmark <name> --exit-policy strict`
5. Hand off to `read-results`

Users never see `--config`, handler class names, or task group identifiers.

**Output artifact:** `run_outcome.json` + `run_summary.json`

---

### Execution: `read-results`
*(Standalone)*

Translates benchmark results into plain English. Not a diagnosis tool (`interpret-run` is for developers) — a communication tool.

- Translate scores: "Your system extracted vendor_name correctly 89% of the time. The weakest field was amount (61%)."
- Show the 2-3 worst-performing samples: input, expected, what the AI actually said.
- One concrete next step.

**Output artifact:** Human-readable summary (no files written)

---

## Skill Chaining

Each skill is standalone and operates on `benchmark.yaml` at the project root. If the file doesn't exist when a skill runs, the skill creates it with only its section populated.

Natural sequence from scratch:

```
define-task          →  task:
define-scoring       →  scoring:
configure-model      →  target:          ← Stage 3: these two are paired,
setup-data           →  data:            ← order is flexible
  └─ synthesize-data →  .data/<name>/train.jsonl   (if no data exists)
──────────────────────────────────────────────────
[benchmark.yaml complete]
run-benchmark        →  smoke → full → run_outcome.json
read-results         →  plain English summary
```

An experienced user can invoke any skill directly ("just update the model in my benchmark") without running the others.

---

## Relationship to Developer Skills

| Developer skill | Second-layer skill | What changes |
|----------------|--------------------|--------------|
| `evaluate` | `run-benchmark` | Users reference benchmark name, not config path |
| `interpret-run` | `read-results` | Output is human summary, not failure diagnosis |
| `add-task` | `define-task` + compiler | No Python, no handler class |
| `add-provider` | `configure-model` | Config only, no interface code |

---

## Files to Create

### Agent skills
```
.agents/skills/define-task/SKILL.md
.agents/skills/define-scoring/SKILL.md
.agents/skills/configure-model/SKILL.md
.agents/skills/setup-data/SKILL.md
.agents/skills/synthesize-data/SKILL.md
.agents/skills/run-benchmark/SKILL.md
.agents/skills/read-results/SKILL.md
```

### Engine code (from `docs/SECOND_LAYER_DESIGN.md` build order)
```
src/benchmark_compiler.py       ~150 lines  (YAML → build_adhoc_task_spec())
src/data_generator.py           ~150 lines  (synthesize-data backend)
benchy eval --benchmark <name>  ~50 lines   (CLI wire-up)
benchy create                   ~300 lines  (interactive wizard, optional)
benchy validate --benchmark     ~100 lines  (pre-flight check, optional)
```

### Key reuse points in existing code
- `src/tasks/registry.py:511` — `build_adhoc_task_spec()` is the compiler's output target
- `src/tasks/common/task_config_schema.py:16` — `TASK_TYPE_SCHEMAS` maps internal type strings to handler classes
- `src/interfaces/generic_api_interface.py:151` — `GenericAPIInterface` handles `target.type: api`
- `src/benchy_cli_eval.py:797` — `--dataset-name` and `--task-type` already exist; `--benchmark` is additive

---

## Engineering Review Blockers (resolve before writing compiler)

1. **Multimodal:** `TASK_TYPE_SCHEMAS` has no `multimodal_structured` entry. Solution: map `input.type: image` to `multimodal_input: true` in the `structured` config — not a separate handler.
2. **`target.type: local`:** means local OpenAI-compatible HTTP endpoint, not a subprocess. Document this explicitly in the compiler.
3. **`benchmark.yaml` location:** project root. `benchy eval --benchmark <name>` reads `./benchmark.yaml` if name matches, or `./benchmark.yaml` directly. Single benchmark per project for v1.
