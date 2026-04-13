# Benchy: Second Layer Design

## The Problem

Benchy's engine is solid. You can benchmark any AI system against any task. But the interface to that engine requires you to be the person who built it. A new user faces: 4 handler classes, Python inheritance, YAML capability declarations, 4 overlapping docs, and a 50-flag CLI. That's not a tool for normal people. It's a tool for contributors.

The second layer fixes this. It sits entirely above the existing engine. The engine doesn't change.

---

## Core User Mental Model

A benchmark has exactly four things:

```
1. TASK        — what the AI is supposed to do (inputs → outputs)
2. SCORING     — how to measure if it did it correctly
3. DATA        — examples to run it against
4. TARGET      — which AI system to evaluate
```

Everything in the second layer maps to one of these four. The user answers four questions and gets a benchmark. The engine handles everything else.

---

## The Benchmark Spec: `benchmark.yaml`

This is the canonical artifact. All second-layer interfaces (wizard, API, future web UI) produce this file. The engine consumes it.

```yaml
benchmark:
  name: invoice-extraction-v1
  description: Extract structured fields from Argentine invoice images

  # 1. TASK — what the AI does
  task:
    type: extraction               # extraction | classification | qa | translation | freeform
    input:
      type: image                  # text | image | document | audio
      description: "Scanned or digital PDF invoice"
    output:
      type: structured             # structured | label | text | score
      fields:
        - name: vendor_name
          type: string
          description: Name of the seller
          required: true
        - name: amount
          type: number
          description: Total invoice amount
          required: true
        - name: date
          type: string
          format: YYYY-MM-DD
          required: false
        - name: currency
          type: string
          description: Currency code (USD, ARS, etc.)
          required: false

  # 2. SCORING — how to measure success
  scoring:
    type: per_field                # binary | per_field | semantic | custom
    # binary:    1 point if ALL fields correct, 0 otherwise
    # per_field: 1 point per correct field (decomposes total score)
    # semantic:  fuzzy match — good for translations, QA, summaries
    # custom:    user provides a Python scoring function (advanced)
    partial_credit: true
    case_sensitive: false
    numeric_tolerance: 0.01        # within 1% counts as correct for numbers

  # 3. DATA — what to run it against
  data:
    source: local                  # local | generate | huggingface
    path: ./data/invoices.jsonl    # for local
    # source: generate
    # count: 50
    # generator_model: gpt-4o
    # seed_description: "Realistic Argentine invoices with varying vendors and amounts"
    # source: huggingface
    # dataset: org/dataset-name
    # split: test

  # 4. TARGET — which AI to evaluate
  target:
    type: api                      # api | model | local
    url: https://api.example.com/extract
    body_template: '{"image": "{{image_path|base64_image}}"}'
    response_path: data
    # type: model
    # provider: openai
    # model: gpt-4o
    # system_prompt: "Extract invoice fields and return as JSON."
    # type: local
    # command: python my_script.py --input {input_path}
```

**Key design decisions:**
- `type` fields use human vocabulary (extraction, classification) not benchy internals (StructuredHandler, MultipleChoiceHandler)
- The compiler maps `task.type + output.type → handler class` automatically
- `scoring.type: per_field` maps to `StructuredHandler + EQS metrics`
- `scoring.type: binary` maps to `ExactMatch`
- `scoring.type: semantic` maps to `FreeformHandler + F1`

---

## The Authoring Interface: `benchy create`

An interactive wizard that produces `benchmark.yaml`. Five questions, smart defaults.

```
╭─────────────────────────────────────────────────────╮
│  benchy create                                       │
│  Define a benchmark for your AI system               │
╰─────────────────────────────────────────────────────╯

? What does your AI do? (describe in plain English)
  › Extracts invoice fields from PDF images

? What type of input does it take?
  ❯ 1. Text (a document, question, or passage)
    2. Image or PDF
    3. Audio
    4. Text + image

? What is the output? How would you grade a correct answer?
  ❯ 1. Extracts specific fields (vendor name, amount, date...)
    2. Classifies into one of several categories
    3. Answers a question in free text
    4. Translates text
    5. Something else — I'll describe it

? Which fields does it extract?
  Add fields one per line. Format: field_name: description
  (Press Enter twice when done)

  vendor_name: Name of the seller
  amount: Total invoice amount
  date: Invoice date
  currency: Currency code

  ✓ 4 fields defined

? How should a correct extraction be scored?
  ❯ 1. Each correct field earns a point (score = correct / total)
    2. All fields must be correct to earn a point (pass/fail)
    3. Approximate matches count (fuzzy — good for messy data)

? Do you have test data?
  ❯ 1. Yes — I have a file (CSV, JSONL, or images)
    2. Yes — but I need to reformat it
    3. No — generate synthetic examples
    4. I'll add data later

  [3] Generating 30 synthetic invoice examples using gpt-4o...
  ✓ Generated 30 examples → .data/invoice-extraction-v1/

? What AI system are you benchmarking?
  ❯ 1. My own API endpoint (HTTP)
    2. A specific model (OpenAI, Anthropic, together, etc.)
    3. A local model (vLLM)
    4. Skip — I'll set this up later

? What is your endpoint URL?
  › https://api.surus.lat/extract

╭─────────────────────────────────────────────────────╮
│  ✓ Benchmark defined                                 │
│                                                      │
│  Name:    invoice-extraction-v1                      │
│  Task:    field extraction from images (4 fields)    │
│  Scoring: per-field                                  │
│  Data:    30 synthetic examples                      │
│  Target:  https://api.surus.lat/extract              │
│                                                      │
│  Files created:                                      │
│    benchmark.yaml                                    │
│    .data/invoice-extraction-v1/ (30 examples)        │
│                                                      │
│  Run a smoke test (5 examples, fast):                │
│  benchy eval --benchmark invoice-extraction-v1 \     │
│              --limit 5 --exit-policy smoke           │
╰─────────────────────────────────────────────────────╯
```

---

## Data Generation

When the user has no data, the second layer generates synthetic examples using a model.

**Inputs:** task description + field schema + count + optional seed description
**Process:**
1. Build a generation prompt from the task spec
2. Call generator model N times
3. Each call produces one `{input, expected}` pair
4. Validate each pair against the field schema
5. Save to `.data/<benchmark-name>/`

**Auto-generated prompt (from spec):**
```
You are generating test data for an AI evaluation benchmark.

Task: Extract invoice fields from PDF images
Fields:
  - vendor_name: Name of the seller
  - amount: Total invoice amount
  - date: Invoice date
  - currency: Currency code

Generate a realistic example. Return JSON:
  "input_description": Detailed description of the invoice image
  "expected": Correct extracted fields as JSON object

Vary: mix vendors, currencies, amounts, date formats.
```

**Honest limitation:** For image-input tasks, synthetic data fills in the `expected` ground truth only. The user still needs real images to pair with it. For text tasks, the input is the text itself — fully synthetic.

---

## Scoring Function Translator

Maps human language to benchy's metric system:

| User says | `scoring.type` | Benchy internal |
|-----------|---------------|-----------------|
| "each correct field earns a point" | `per_field` | `StructuredHandler` + `EQS` |
| "all fields must be correct" | `binary` | `ExactMatch` |
| "approximate matches count" | `semantic` | `FreeformHandler` + `F1Score` |
| "I'll define my own" | `custom` | user Python function |

For `per_field`, the score per example:
```
score = (correctly extracted fields) / (total required fields)
```

---

## Compiler: `benchmark.yaml` → Engine

Reads `benchmark.yaml`, returns a `FormatHandler` instance. Plugs directly into `BenchmarkRunner` via the existing protocol. ~200 lines.

**Mapping rules:**

```
task.type + output.type            → handler
────────────────────────────────────────────
extraction + structured            → StructuredHandler
classification + label             → MultipleChoiceHandler
qa + text                          → FreeformHandler
translation + text                 → FreeformHandler
extraction + structured + image    → MultimodalStructuredHandler

scoring.type                       → metrics
────────────────────────────────────────────
per_field                          → metrics_config (EQS, field-level F1)
binary                             → ExactMatch
semantic                           → [F1Score, ExactMatch]

target.type                        → interface
────────────────────────────────────────────
api                                → GenericAPIInterface
model                              → OpenAIInterface
local                              → OpenAIInterface (local endpoint)
```

**Integration point:**

```
benchmark.yaml
     ↓
  compiler.py (new ~200 lines)
     ↓
  FormatHandler instance         ← existing protocol, unchanged
     ↓
  BenchmarkRunner (unchanged)
     ↓
  run_outcome.json (unchanged)
```

---

## CLI Changes

Two new commands, no changes to existing ones:

```bash
benchy create                              # wizard → benchmark.yaml
benchy eval --benchmark <name>             # compiles + runs
benchy validate --benchmark <name>         # pre-flight check (optional, add later)
```

`--benchmark` compiles `benchmark.yaml` then delegates to the existing eval pipeline. `--config` stays for contributor tasks. Both paths converge at `BenchmarkRunner`.

---

## What Doesn't Change

- `BenchmarkRunner`, handlers, interfaces, metrics — untouched
- Contributor workflow (Python handler classes, metadata.yaml) — still works
- AGENTS.md and the skill system — still the agent interface
- Existing task library (Spanish, Portuguese, document extraction) — still there

The second layer is a new front door to the same building.

---

## Build Order

1. `docs/SECOND_LAYER_DESIGN.md` — this doc (done)
2. `src/benchmark_compiler.py` — YAML → handler. ~200 lines.
3. `benchy eval --benchmark` — wire compiler into CLI. ~50 lines.
4. `benchy create` wizard — interactive authoring. ~300 lines.
5. `src/data_generator.py` — synthetic data from task spec. ~150 lines.
6. `benchy validate --benchmark` — pre-flight schema check. ~100 lines.

**Total: ~800 lines of new code. Zero changes to the engine.**
