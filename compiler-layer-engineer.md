# Second Layer: Implementation Complete

## What Was Built

The second layer sits above the existing engine and exposes benchmarking to non-developer users through a `benchmark.yaml` spec file and a set of agent skills. Zero changes to the engine.

---

## Files Created

### `src/benchmark_compiler.py`
Translates `benchmark.yaml` into engine-ready configuration structures.

- `compile_benchmark(path)` — reads the spec and returns a `CompiledBenchmark` dataclass with `internal_task_type`, `dataset_config`, `target_config`, `system_prompt`
- `validate_benchmark_yaml(path)` — pre-flight check; returns a list of human-readable error messages
- Maps user vocabulary (`extraction`, `classification`, `qa`, `translation`) to internal handler types (`structured`, `classification`, `freeform`)
- Builds JSON Schema from field definitions in `task.output.fields`
- Maps `target.type` (`api` / `model` / `local`) to provider config dicts the engine already understands

### `src/data_generator.py`
Generates synthetic `{text, expected}` pairs from a task spec using an LLM.

- `generate_data(benchmark_yaml_path, count, output_dir)` — async parallel generation, validates each sample against the field schema, saves to `.data/<benchmark-name>/train.jsonl`
- Hard blocks image/document input tasks with a clear message
- Progress output: `Generated 12/30...`

### `src/benchy_create.py`
Interactive four-question wizard that produces `benchmark.yaml`.

- Collects task definition, scoring method, target system, and data source
- Runs data generation inline if the user picks "synthesize"
- Prints a summary box with next-step command on completion
- Invoked via `benchy create`

### Changes to `src/benchy_cli_eval.py`
- Added `--benchmark YAML_PATH` argument to `add_eval_arguments`
- Added `_apply_benchmark_to_args(args)` — compiles `benchmark.yaml` and injects equivalent CLI args into the `args` namespace before the existing `run_eval` flow runs. No engine changes.

### Changes to `src/benchy_cli.py`
- Added `benchy create` subcommand → calls `run_create_wizard()`
- Added `benchy validate` subcommand → calls `validate_benchmark_yaml()` and prints errors or a ready message

---

## New CLI Commands

```bash
# Interactive wizard — produces benchmark.yaml
benchy create
benchy create --output my-benchmark.yaml

# Pre-flight validation
benchy validate
benchy validate --benchmark my-benchmark.yaml

# Compile and run (second-layer entry point)
benchy eval --benchmark benchmark.yaml --limit 5 --exit-policy smoke
benchy eval --benchmark benchmark.yaml --exit-policy strict
```

---

## Agent Skills Created (7 files)

| Skill | Path | Stage | Purpose |
|-------|------|-------|---------|
| `define-task` | `.agents/skills/define-task/SKILL.md` | 1 | Collect task definition → `task:` section |
| `define-scoring` | `.agents/skills/define-scoring/SKILL.md` | 2 | Choose scoring method → `scoring:` section |
| `configure-model` | `.agents/skills/configure-model/SKILL.md` | 3a | Specify AI system → `target:` section |
| `setup-data` | `.agents/skills/setup-data/SKILL.md` | 3b | Supply test data → `data:` section |
| `synthesize-data` | `.agents/skills/synthesize-data/SKILL.md` | utility | Generate synthetic examples from task spec |
| `run-benchmark` | `.agents/skills/run-benchmark/SKILL.md` | exec | Validate + smoke + full run from `benchmark.yaml` |
| `read-results` | `.agents/skills/read-results/SKILL.md` | exec | Translate results into plain English for non-developers |

---

## benchmark.yaml Format

```yaml
benchmark:
  name: invoice-extraction-v1
  description: Extract fields from Argentine invoices

  task:
    type: extraction             # extraction | classification | qa | translation | freeform
    input:
      type: image                # text | image | document
      description: "Scanned invoice PDF"
    output:
      type: structured
      fields:
        - name: vendor_name
          type: string
          required: true
        - name: amount
          type: number
          required: true

  scoring:
    type: per_field              # per_field | binary | semantic
    case_sensitive: false
    numeric_tolerance: 0.01

  data:
    source: local                # local | generate | huggingface
    path: .data/invoices/train.jsonl

  target:
    type: api                    # api | model | local
    url: https://api.example.com/extract
    body_template: '{"image": "{{image_path|base64_image}}"}'
    response_path: data
```

---

## How It Connects to the Engine

```
benchmark.yaml
     ↓
benchmark_compiler.py       compile_benchmark()
     ↓
args namespace injection     _apply_benchmark_to_args()
     ↓
_build_cli_only_config()     existing path, unchanged
     ↓
BenchmarkRunner              existing engine, unchanged
     ↓
run_outcome.json             existing output, unchanged
```

The compiler translates the user-facing spec into the same `args` fields that a developer would pass via CLI flags. The engine never knows it came from a `benchmark.yaml`.
