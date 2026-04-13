# Benchy Second Layer — Walkthrough

You built an AI system. You want to know how well it works. This is the full path from nothing to results.

---

## What you need before starting

- Benchy installed and `.env` configured with your API keys
- An AI system to evaluate — an HTTP endpoint, a model, or a local server
- Optionally: test data (CSV or JSONL). If you don't have any, benchy can generate it.

---

## The short version

```bash
benchy create                          # answer 4 questions → benchmarks/my-benchmark.yaml
benchy validate                        # check the spec is valid
benchy eval --limit 5 --exit-policy smoke   # smoke test (5 examples)
benchy eval --exit-policy strict            # full run
```

That's it. Everything below is the detailed version of those four commands.

---

## Scenario 1: Invoice extraction API

You have an API that extracts fields from invoice images. You want to know how accurate it is.

### Step 1 — Create the benchmark

```
$ benchy create
```

The wizard asks four groups of questions:

```
╭──────────────────────────────────────────────╮
│  benchy create                                │
│  Define a benchmark for your AI system        │
│  Four steps. No code required.                │
╰──────────────────────────────────────────────╯

Benchmark name (used for output folders, no spaces) [my-benchmark]: invoice-extraction
Short description (one sentence): Extract vendor name, amount and date from invoices

── Stage 1: Define the task ──────────────────────────────────
What does your AI do? (plain English): Extracts structured fields from invoice PDFs

What type of input does it take?
  1. Text (a document, question, or passage)
  2. Image or PDF  ← pick this
  3. Text + image

What does it produce?
  1. Extracts specific fields (name, amount, date…)  ← pick this
  2. Classifies into one of several categories
  3. Answers a question in free text
  4. Translates text
  5. Other — I'll describe it

List the fields your AI should extract.
Format: field_name: description (one per line, blank line to finish)
  › vendor_name: Name of the seller
  › amount: Total invoice amount
  › date: Invoice date in YYYY-MM-DD format
  › currency: Currency code
  ›
  ✓ 4 fields defined

── Stage 2: Define scoring ───────────────────────────────────
How should a correct output be scored?
  1. Each correct field earns a point  ← pick this
  2. All fields must be correct to earn a point (pass/fail)
  3. Approximate matches count (fuzzy)

  Numeric tolerance (e.g. 0.01): 0.01
  Case-sensitive comparison? [y/N]: n

── Stage 3b: Set up data ─────────────────────────────────────
Do you have test data?
  1. I have a file (CSV or JSONL)
  2. I have data but need to reformat it
  3. No data — generate synthetic examples
  4. Skip — I'll set this up later

  → You picked image input. Synthesis can't produce real images.
  → Use option 1 or 2 to supply your own images.

  (pick 1) Path to your file: .data/invoices/train.jsonl

── Stage 3a: Configure target ────────────────────────────────
What AI system are you benchmarking?
  1. My own API endpoint (HTTP)  ← pick this
  2. A specific model (OpenAI, Anthropic, Together, etc.)
  3. A local model (OpenAI-compatible server)
  4. Skip — I'll set this up later

  Endpoint URL: https://api.example.com/v1/extract
  Body template [leave blank for default]: {"image": "{{image_path|base64_image}}"}
  Response path: data

╭──────────────────────────────────────────────────────╮
│  ✓ Benchmark defined                                  │
│                                                       │
│  Name:     invoice-extraction                         │
│  Task:     extraction (4 fields)                      │
│  Scoring:  per-field                                  │
│  Data:     .data/invoices/train.jsonl                 │
│  Target:   https://api.example.com/v1/extract         │
│                                                       │
│  File:     benchmarks/invoice-extraction.yaml         │
│                                                       │
│  Next — smoke test (5 examples):                      │
│    benchy eval --benchmark benchmarks/invoice-        │
│    extraction.yaml --limit 5 --exit-policy smoke      │
╰──────────────────────────────────────────────────────╯
```

The wizard wrote `benchmarks/invoice-extraction.yaml`. Look at it:

```yaml
benchmark:
  name: invoice-extraction
  description: Extract vendor name, amount and date from invoices

  task:
    type: extraction
    input:
      type: image
      description: Extracts structured fields from invoice PDFs
    output:
      type: structured
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
          description: Currency code
          required: false

  scoring:
    type: per_field
    case_sensitive: false
    numeric_tolerance: 0.01

  data:
    source: local
    path: .data/invoices/train.jsonl

  target:
    type: api
    url: https://api.example.com/v1/extract
    body_template: '{"image": "{{image_path|base64_image}}"}'
    response_path: data
```

You can edit this file directly at any time. The wizard is just a shortcut for writing it.

---

### Step 2 — Validate

```
$ benchy validate
✓ benchmarks/invoice-extraction.yaml is valid and ready to run.

  Run a smoke test:
  benchy eval --benchmark benchmarks/invoice-extraction.yaml --limit 5 --exit-policy smoke
```

If something is missing or wrong:

```
✗ benchmarks/invoice-extraction.yaml has 1 error(s):

  • target.body_template is required when target.type is 'api'
```

Fix it in the YAML, re-run validate, repeat until clean.

---

### Step 3 — Smoke test (5 examples)

Always run a smoke test first. It catches connectivity issues, wrong response formats, and dataset problems before spending time on the full run.

```
$ benchy eval --benchmark benchmarks/invoice-extraction.yaml --limit 5 --exit-policy smoke
```

Check the result:

```
$ cat outputs/benchmark_outputs/<run_id>/invoice-extraction/run_outcome.json
```

You want:
```json
{
  "status": "passed",
  "counts": {
    "passed_tasks": 1,
    "failed_tasks": 0,
    "error_tasks": 0,
    "no_samples_tasks": 0
  }
}
```

Common smoke failures and what they mean:

| What you see | What it means | Fix |
|---|---|---|
| `no_samples` | Dataset not found | Check `data.path` in the spec |
| `connectivity_error` | API unreachable | Check `target.url` and your network |
| `all_invalid_responses` | Wrong response format | Check `target.response_path` |
| `json_parse_error` | Response isn't JSON | Check what your API actually returns |

---

### Step 4 — Full run

Smoke passed. Run the full dataset.

```
$ benchy eval --benchmark benchmarks/invoice-extraction.yaml --exit-policy strict
```

Results land in `outputs/benchmark_outputs/<run_id>/invoice-extraction/`.

---

### Step 5 — Read the results

```
outputs/benchmark_outputs/<run_id>/invoice-extraction/
├── run_outcome.json      ← overall pass/fail and counts
├── run_summary.json      ← per-field scores
└── invoice-extraction/
    └── main/
        ├── main_metrics.json    ← numeric scores
        └── main_samples.json   ← individual predictions
```

`run_summary.json` gives you the top-line numbers:
```json
{
  "invoice-extraction": {
    "main": {
      "vendor_name_f1": 0.94,
      "amount_f1": 0.61,
      "date_f1": 0.88,
      "currency_f1": 0.79,
      "overall_score": 0.81
    }
  }
}
```

`main_samples.json` shows individual predictions — useful for understanding where it fails:
```json
{
  "samples": [
    {
      "id": "42",
      "input": "...",
      "expected": {"vendor_name": "ACME Corp", "amount": 1250.00, ...},
      "prediction": {"vendor_name": "ACME Corp", "amount": "1.250,00", ...},
      "score": 0.75
    }
  ]
}
```

---

## Scenario 2: No data — generate synthetic examples

You're benchmarking a QA system but don't have a test set yet.

### Step 1 — Create the benchmark

```
$ benchy create
```

When asked about data, choose option 3 (generate):

```
Do you have test data?
  3. No data — generate synthetic examples  ← pick this

  How many examples? [30]: 50
  Describe the kind of examples to generate: Customer support questions about billing and account management
```

The wizard generates 50 examples on the spot and writes them to `.data/qa-support/train.jsonl`. The spec's `data:` section is updated to point to the generated file.

The resulting spec:

```yaml
  task:
    type: qa
    input:
      type: text
      description: Customer support QA system
    output:
      type: text

  scoring:
    type: semantic

  data:
    source: local
    path: .data/qa-support/train.jsonl

  target:
    type: model
    provider: openai
    model: gpt-4o-mini
    system_prompt: "You are a helpful customer support agent."
```

From here the flow is the same: validate → smoke → full run.

---

## Scenario 3: Multiple benchmarks in one project

You're comparing two systems — your API and GPT-4o — on the same extraction task. Plus you have a separate translation benchmark.

```
benchmarks/
  invoice-api.yaml         # your extraction API
  invoice-gpt4o.yaml       # GPT-4o on the same task
  translation-v2.yaml      # separate translation benchmark
```

List them:

```
$ benchy benchmarks
  invoice-api       (benchmarks/invoice-api.yaml)       Invoice extraction — custom API
  invoice-gpt4o     (benchmarks/invoice-gpt4o.yaml)     Invoice extraction — GPT-4o baseline
  translation-v2    (benchmarks/translation-v2.yaml)    ES→EN translation quality
```

With multiple specs, always pass `--benchmark` explicitly:

```bash
# Run one
benchy eval --benchmark benchmarks/invoice-api.yaml --limit 5 --exit-policy smoke

# Run another
benchy eval --benchmark benchmarks/invoice-gpt4o.yaml --limit 5 --exit-policy smoke

# Validate one
benchy validate --benchmark benchmarks/translation-v2.yaml
```

If you try to run without specifying:

```
$ benchy eval --limit 5
Multiple benchmark specs found. Pass --benchmark to choose one:
  benchy eval --benchmark benchmarks/invoice-api.yaml
  benchy eval --benchmark benchmarks/invoice-gpt4o.yaml
  benchy eval --benchmark benchmarks/translation-v2.yaml
```

---

## Editing benchmark.yaml directly

The wizard is optional. You can write or edit the spec by hand at any time — it's just YAML. The full format:

```yaml
benchmark:
  name: my-benchmark
  description: One line description

  task:
    type: extraction | classification | qa | translation | freeform
    input:
      type: text | image | document
      description: "What the input looks like"
    output:
      type: structured | label | text
      fields:                          # only for structured output
        - name: field_name
          type: string | number | integer | boolean
          description: What this field contains
          required: true | false

  scoring:
    type: per_field | binary | semantic
    case_sensitive: true | false       # optional
    numeric_tolerance: 0.01            # optional, for number fields

  data:
    source: local | generate | huggingface
    path: .data/<name>/train.jsonl     # for local
    count: 30                          # for generate
    seed_description: "..."            # for generate, optional
    dataset: org/dataset-name         # for huggingface
    split: test                        # for huggingface

  target:
    # Custom API
    type: api
    url: https://your-endpoint.com/extract
    body_template: '{"input": "{{text}}"}'
    response_path: data

    # Named model
    type: model
    provider: openai | anthropic | together | google
    model: gpt-4o
    system_prompt: "Optional system prompt"

    # Local server
    type: local
    url: http://localhost:8000/v1
    model: meta-llama/Llama-3.1-8B-Instruct
```

---

## Iterating

A typical iteration loop:

1. Run the benchmark, check field scores in `run_summary.json`
2. Find the weakest field — look at its samples in `main_samples.json`
3. Understand the failure pattern (formatting issue? missing field? wrong value?)
4. Improve your system or adjust the system prompt in `target.system_prompt`
5. Re-run with a new `--run-id`

Each run gets its own directory under `outputs/benchmark_outputs/`. Old runs are never overwritten.

---

## Quick reference

| Command | What it does |
|---|---|
| `benchy create` | Interactive wizard → `benchmarks/<name>.yaml` |
| `benchy benchmarks` | List all specs in this project |
| `benchy validate` | Pre-flight check (auto-discovers if one spec exists) |
| `benchy validate --benchmark <path>` | Validate a specific spec |
| `benchy eval --limit 5 --exit-policy smoke` | Smoke test (auto-discovers) |
| `benchy eval --benchmark <path> --limit 5 --exit-policy smoke` | Smoke test a specific spec |
| `benchy eval --benchmark <path> --exit-policy strict` | Full run |
| `benchy eval --benchmark <path> --run-id my-run` | Named run (resumable) |
