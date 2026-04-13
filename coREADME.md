# Benchy Second Layer — World Model

Benchy has two layers. The first is the engine: handlers, interfaces, metrics, runners. It is solid and contributor-facing. The second layer is this: a human-vocabulary interface that sits on top of the engine and exposes benchmarking to people who built an AI system and want to know how well it works.

The second layer does not change the engine. It is a translation system.

---

## The Core Idea

A benchmark has exactly four things. Always four, never more:

```
TASK      — what the AI is supposed to do
SCORING   — how to measure if it did it correctly
DATA      — the examples to run it against
TARGET    — which AI system is being evaluated
```

Everything in the second layer maps to one of these four. `benchmark.yaml` has four sections. `benchy create` asks four groups of questions. The compiler translates each section into the engine's vocabulary. That's the whole model.

---

## The Compiler

`src/benchmark_compiler.py` is the centerpiece of the second layer. It is a translator.

The engine speaks in handler classes, dataset configs, provider configs, and metrics manifests. Users speak in extractions, pass/fail, their API endpoint, and their CSV. The compiler bridges the two.

```
benchmark.yaml              user vocabulary
      ↓
benchmark_compiler.py       translation
      ↓
dataset_config              }
target_config               } engine vocabulary
internal_task_type          }
      ↓
BenchmarkRunner             unchanged engine
      ↓
run_outcome.json            unchanged output
```

The engine never knows a `benchmark.yaml` existed. By the time the compiler is done, everything looks like a developer-authored config.

### What the compiler does

**1. Resolves task type.**
Maps the user's intent (`extraction`, `classification`, `qa`, `translation`) and output shape (`structured`, `label`, `text`) to the internal handler type (`structured`, `classification`, `freeform`). Image input doesn't change the handler — it sets `multimodal_input: true` on the dataset config.

| `task.type` + `output.type` | Internal type |
|-----------------------------|---------------|
| `extraction` + `structured` | `structured` |
| `classification` + `label` | `classification` |
| `qa` / `translation` / `freeform` + `text` | `freeform` |
| any of the above + `input.type: image` | same + `multimodal_input: true` |

**2. Builds the JSON Schema.**
For extraction tasks, the user lists fields in plain English. The compiler turns that into a proper JSON Schema object and embeds it in the dataset config as `schema_json`. The handler picks it up without any further work.

**3. Translates scoring.**
Maps the user's grading choice to the engine's metric system.

| `scoring.type` | Engine behavior |
|----------------|----------------|
| `per_field` | field-level F1 / EQS — one score per extracted field |
| `binary` | `ExactMatch` — all correct or zero |
| `semantic` | `F1Score` + `ExactMatch` — fuzzy, good for QA and translation |

**4. Resolves the data source.**
Maps `data.source` (`local`, `generate`, `huggingface`) to a dataset config the engine can load. For `generate`, the data is expected to already exist in `.data/<name>/` from a prior `synthesize-data` run.

**5. Builds the provider config.**
Maps the target to a provider type and config dict.

| `target.type` | Engine interface |
|---------------|----------------|
| `api` | `GenericAPIInterface` — custom HTTP, template-based |
| `model` | `OpenAIInterface` — named cloud provider |
| `local` | `OpenAIInterface` — local base_url |

The result is a `CompiledBenchmark` dataclass. `benchy eval --benchmark` injects its fields into the same `args` namespace the rest of the eval pipeline already reads. Nothing downstream changes.

---

## benchmark.yaml

The spec file. One per benchmark — a project typically has several. The recommended layout is a `benchmarks/` directory at the project root:

```
benchmarks/
  invoice-extraction.yaml
  translation-v2.yaml
  classification-pilot.yaml
```

A single `benchmark.yaml` at the root also works and is auto-discovered — convenient for simple projects. When exactly one spec exists anywhere, all commands find it automatically. When multiple exist, pass `--benchmark <path>` explicitly.

Every second-layer interface — the wizard, the CLI, an agent — produces and consumes this file.

```yaml
benchmark:
  name: invoice-extraction-v1
  description: Extract structured fields from Argentine invoice images

  task:
    type: extraction
    input:
      type: image
      description: "Scanned or digital PDF invoice"
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

  scoring:
    type: per_field
    case_sensitive: false
    numeric_tolerance: 0.01

  data:
    source: local
    path: .data/invoices/train.jsonl

  target:
    type: api
    url: https://api.example.com/extract
    body_template: '{"image": "{{image_path|base64_image}}"}'
    response_path: data
```

The key design decision: every field uses the vocabulary of the problem domain, not benchy's internals. `extraction`, not `StructuredHandler`. `per_field`, not `MetricsCalculator`. The compiler owns the mapping.

---

## The Authoring Surface

Four CLI commands for the second layer:

### `benchy benchmarks` — list

Shows all specs in this project. Scans `benchmark.yaml` at root and `benchmarks/*.yaml`.

```bash
benchy benchmarks
benchy benchmarks --json
```

```
  invoice-extraction  (benchmarks/invoice-extraction.yaml)  Extract fields from Argentine invoices
  translation-v2      (benchmarks/translation-v2.yaml)      Spanish → English translation quality
```

### `benchy create` — wizard

Guides the user through the four sections interactively. Writes to `benchmarks/<name>.yaml` by default.

```bash
benchy create
benchy create --output benchmarks/my-benchmark.yaml
benchy create --output benchmark.yaml   # single-file layout
```

### `benchy validate` — pre-flight

Validates a spec before running. Auto-discovers if exactly one spec exists. Returns human-readable errors, not stack traces.

```bash
benchy validate                                           # auto-discover
benchy validate --benchmark benchmarks/my-benchmark.yaml
```

```
✗ benchmarks/my-benchmark.yaml has 2 error(s):

  • task.output.fields is required when task.output.type is 'structured'
  • target.url is required when target.type is 'api'
```

### `benchy eval --benchmark` — run

Compiles and runs. Auto-discovers if exactly one spec exists. All existing eval flags (`--limit`, `--exit-policy`, `--run-id`) still work.

```bash
# Auto-discover (single spec or benchmark.yaml at root)
benchy eval --limit 5 --exit-policy smoke

# Explicit path (required when multiple specs exist)
benchy eval --benchmark benchmarks/invoice-extraction.yaml --limit 5 --exit-policy smoke
benchy eval --benchmark benchmarks/invoice-extraction.yaml --exit-policy strict
```

---

## Data Generation

`src/data_generator.py` handles the case where the user has no test data. It reads the task spec, builds a generation prompt from the field definitions, calls a model in parallel for each example, validates each result against the schema, and saves to `.data/<name>/train.jsonl`.

Set `data.source: generate` in `benchmark.yaml`, or call directly:

```python
from src.data_generator import generate_data
generate_data("benchmark.yaml", count=30)
# → .data/invoice-extraction-v1/train.jsonl
```

One hard constraint: image and document input tasks cannot be synthesized. The generator can produce expected outputs but not real images. It raises a clear error and points the user to their data instead.

---

## Agent Skills

Seven SKILL.md files encode the canonical agent workflow for each stage of the second layer. Agents should consult these before acting — they are the authoritative guide to what to ask, what to produce, and what to write into `benchmark.yaml`.

```
.agents/skills/
├── define-task/        Stage 1 — task: section
├── define-scoring/     Stage 2 — scoring: section
├── configure-model/    Stage 3a — target: section
├── setup-data/         Stage 3b — data: section
├── synthesize-data/    utility — generate examples from spec
├── run-benchmark/      execution — validate + smoke + full run
└── read-results/       communication — plain-English result summary
```

Each skill is standalone and operates on a benchmark spec file. When the project has a single spec, skills auto-discover it. When multiple specs exist, pass the path explicitly. They can be run in any order. If the file doesn't exist when a skill runs, the skill creates it with only its section populated.

---

## Python API

```python
from src.benchmark_compiler import compile_benchmark, validate_benchmark_yaml

errors = validate_benchmark_yaml("benchmark.yaml")  # → [] if valid

compiled = compile_benchmark("benchmark.yaml")
compiled.name                # "invoice-extraction-v1"
compiled.internal_task_type  # "structured"
compiled.dataset_config      # ready for BenchmarkRunner
compiled.target_config       # provider/model config dict
compiled.system_prompt       # optional
compiled.scoring             # raw scoring section
```

---

## Boundary with the Engine

The second layer touches three things in the existing codebase:

| Touch point | What happens |
|-------------|-------------|
| `benchy_cli_eval.py` | `--benchmark` flag added; `_apply_benchmark_to_args()` injects compiled fields into `args` |
| `benchy_cli.py` | `benchy benchmarks`, `benchy create`, and `benchy validate` subcommands added |
| Nothing else | Engine, handlers, interfaces, metrics, runners — all unchanged |

The compiler produces `dataset_config` and `target_config` in the same shape that `_build_dataset_config_from_args()` and `_build_cli_only_config()` would produce from CLI flags. The injection point is a single function call at the top of `run_eval`. After that, the code path is identical.

---

## What Does Not Change

The contributor workflow — Python handler classes, `metadata.yaml`, the `add-task` and `add-provider` developer skills — is entirely untouched. The second layer is a new entry point, not a replacement. Both paths reach the same engine.
