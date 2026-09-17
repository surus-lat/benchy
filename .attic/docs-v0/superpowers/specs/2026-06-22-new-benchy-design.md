# New Benchy — Redesign Spec

**Status:** Draft, on branch `REF/new-benchy`, awaiting user approval.
**North star:** `VISION.md` (this branch).
**Companion docs:** `2026-06-22-new-benchy-salvage-audit.md`, `../plans/2026-06-22-new-benchy-execution.md`.

## 1. The two beliefs benchy is built on

From `VISION.md`, restated as design constraints:

1. **The primitive is the AI-system, not the model.** An AI-system is any
   program that performs a task: a raw model, a node (model + prompt), a
   workflow (composed models), or an agent (models + tools + loop).
   Everything downstream — how we run inference, how we swap models, how
   we score — must be blind to which of the four shapes it is talking to.

2. **Benchy creates benchmarks; it does not run other people's
   benchmarks.** A benchmark is the *first step* of AI development — the
   translation layer between a business problem and the AI-system that
   solves it. Authoring a benchmark must feel like designing an exam:
   what is the task, what does correctness mean, what does the data look
   like, what system are we grading.

Every design decision below is checked against these two beliefs.

## 2. The four modules

Benchy has exactly four first-class modules, one per authoring concern.
They are peers. No module owns another.

```
Task           Scoring        System         Data
──────────    ──────────    ──────────    ──────────
input/out     rubric as     model|node    samples that
schema        symbolic      workflow|     satisfy the
(what)        program       agent (how)   task schema
                (grade)                   (evidence)
```

### 2.1 `benchy.task` — the exam contract

A `Task` is the input/output schema of the program the AI-system must
implement. It is *only* schema and prompt scaffolding — never data,
never scoring, never inference details.

```python
task = benchy.Task(
    input=ImageIn(),
    output=InvoiceExtraction(),        # a pydantic model or JSON schema
    domain="invoices",
    language="es-AR",
)
```

A Task answers: *what shape does a solution have?* Nothing more.

### 2.2 `benchy.scoring` — the rubric as symbolic program

A `Scorer` is a symbolic expression, not a class-per-rubric. It is
composable (small pieces combine into big rubrics), introspectable
(reading it tells you what the benchmark *means*), and transformable
(operators produce new scorers from old ones).

Concept locked in the earlier brainstorm:

```python
default   = field_wise(fields=INVOICE_FIELDS, per_field=exact_match())
strict    = binary(default)
weighted  = field_wise_weighted(
    fields=INVOICE_FIELDS,
    per_field=exact_match(),
    weights={"vendor": 1, "total": 3, "due_date": 2},
)
```

Every node is a `Scorer`. Every scorer exposes:

- `evaluate(prediction, expected, sample) -> {value, breakdown}` — per
  sample, rich.
- `fitness(prediction, expected, sample) -> float` — per sample,
  scalar. This is the single number an optimizer will consume.
- `aggregate(samples) -> dict` — over the run.
- `repr` that round-trips: reading the string reconstructs the scorer.

Primitives ship for atomic comparators (`exact_match`, `f1_token`,
`wer`, `cer`, `embedding_sim`), for structural decomposition
(`field_wise`, `field_wise_weighted`), and for generic transforms
(`binary`, `threshold`, `restrict`, `mean`). Adding a family is a
contributor activity; authoring a rubric is not.

### 2.3 `benchy.system` — the AI-system as opaque program

A `System` implements one method:

```python
class System(Protocol):
    capabilities: SystemCapabilities

    async def run(self, sample: dict) -> Prediction: ...
```

Whether the system is a raw HF model, an OpenAI endpoint, a DSPy
program, a LangGraph agent — the contract is the same. Benchy's job at
this layer is to hide the plumbing (transports, venvs, remote/local,
architecture-specific loaders) so the author writes:

```python
system = benchy.System.load("openai:gpt-5-mini")
# or
system = benchy.System.load("hf:Qwen/Qwen3-ASR-0.6B")
# or
system = benchy.System.load("./my_agent.py:Agent")
```

Four `System` shapes ship in-tree (model, node, workflow, agent), each
one loadable from one of a small number of urls. Everything else is
plumbing.

### 2.4 `benchy.data` — the evidence

A `Data` object is a stream of samples that satisfy the Task's input
schema and carry the expected output the Scorer will grade against.
Sources: local files, Hugging Face, curated collections, synthetic
generators. Concerns: loading, caching, validation against the Task
schema, splits.

```python
data = benchy.Data(
    source="hf:google/fleurs",
    subset="es_419",
    split="test",
    schema=task.schema,
)
```

Loading and caching engineering hides here so authoring stays clean.

## 3. Composition — the Benchmark object

The four peers compose into a `Benchmark`. That composition is the only
new object in benchy that ties the peers together:

```python
bench = benchy.Benchmark(
    task=task,
    scoring=weighted,
    data=data,
    system=system,
)

result = await bench.run(limit=200)
```

A `Benchmark` is:

- **Runnable.** `await bench.run(...)` executes the system over the
  data and scores each prediction.
- **Exportable as a loss function.** `bench.as_loss()` returns a
  callable `(system, examples) -> float` suitable for a Software-3.0
  optimizer (DSPy, TextGrad, GEPA-style). This is the "new loss
  function" the vision demands, and it drops out of the design
  naturally: task fixes the contract, data fixes the evidence, scorer
  fixes the fitness — a system is the only free variable, so a
  benchmark is already almost a loss function.
- **Serializable.** A Benchmark round-trips to a small YAML: four
  fields, one per peer. No hidden state.

## 4. Ontology

Vision: `/<task?>/<domain?>/<language?>`. This becomes the *organizing*
structure — how benchmarks are named, where their files live, how the
registry looks them up.

- Filesystem: `benchmarks/<task>/<domain>/<language>/benchmark.yaml`
- Registry key: `image_extraction/invoices/es-AR`
- Any level may be absent: `transcription`, `transcription/fleurs`,
  `transcription/fleurs/pt-BR` are all valid keys with the missing
  levels implied.

The task type in the ontology (e.g. `image_extraction`,
`transcription`) is not a magic string — it references a Task shape
declared in `benchy.task`. Domain and language are pure organizational
labels.

## 5. Authoring workflow

The dev's mental model, mirrored 1:1 by CLI and SDK:

```
1. Define the task     (or reuse an existing task shape)
2. Define the scoring  (compose primitives; transform a default)
3. Curate the data     (point at HF / local / synth; validate against schema)
4. Point at a system   (model | node | workflow | agent url)
5. Run                 (or export as loss and hand to an optimizer)
```

CLI form (illustrative):

```
$ benchy new image_extraction/invoices/es-AR
# scaffolds a benchmark.yaml + a scoring.py stub

$ benchy run image_extraction/invoices/es-AR --system openai:gpt-5-mini

$ benchy export-loss image_extraction/invoices/es-AR --to dspy
```

SDK form (illustrative):

```python
import benchy as b

task     = b.Task.load("image_extraction/invoices/es-AR")
scoring  = b.field_wise_weighted(fields=INVOICE_FIELDS, weights=WEIGHTS)
data     = b.Data.load("./data/invoices/*.pdf", schema=task.schema)
system   = b.System.load("openai:gpt-5-mini")

result   = await b.Benchmark(task, scoring, data, system).run()
```

No mention of interfaces, adapters, providers, pipelines, checkpoints,
runners, probes. Those are plumbing under the hood.

## 6. Reference example

`image_extraction/invoices/es-AR` — end-to-end, so the design is
falsifiable:

- **Task.** Input = `Image`. Output = `InvoiceExtraction` (pydantic
  with `vendor: str`, `total: Decimal`, `due_date: date`, `line_items:
  list[LineItem]`, ...).
- **Scoring.** Default `field_wise(fields=[vendor, total, due_date,
  line_items], per_field=exact_match())`. Variant
  `field_wise_weighted(..., weights={total: 3, due_date: 2, ...})` for
  a business-weighted view. Variant `binary(default)` for a perfect-
  extraction leaderboard.
- **Data.** Curated set of ~200 real Argentine invoice PDFs (or
  images), each with a ground-truth `InvoiceExtraction` JSON.
- **System.** Any VLM loadable via `System.load("...")` — a raw model,
  a `model + prompt` node, or an agent that pre-processes the image.

This is the sanity check for whether the four modules and their
composition are actually enough.

## 7. Salvage — what survives from current benchy

Concise summary; full triage in
`2026-06-22-new-benchy-salvage-audit.md`.

**Keeps (small, reusable, primitives-shaped):**
- `src/tasks/common/metrics.py` — WER, CER, F1, ExactMatch, MSE,
  Pearson, MultipleChoiceAccuracy. These become primitives in
  `benchy.scoring`.
- `src/tasks/common/image_metrics.py` — mask/IoU utilities for image
  tasks.
- `src/tasks/structured_extraction/utils/partial_matching.py` +
  `structured_metrics_calculator.py` — the logic behind `field_wise`
  and partial credit.
- The three custom adapters (`voxtral_chat`, `qwen3_asr_chat`,
  `canary_nemo`) — become `System` implementations for those model
  families.
- Dataset caching primitives in `common/utils/dataset_utils.py` (HF
  download + JSONL cache) — inform `benchy.data`.
- The FLEURS transcription assets and structured-extraction sample
  datasets — data lands under `benchmarks/`.

**Nukes (path-dependent baggage from the glue-frameworks era):**
- `src/pipeline.py` (Prefect flow), `src/engine/benchmark_runner.py`,
  `src/benchy_cli_eval.py`, `src/probe/`.
- `src/interfaces/*` — replaced by `benchy.system`.
- `src/adapters/base.py` — replaced by `benchy.system.System`.
- `src/inference/vllm_*` — one flavor of transport; folded into a
  generic `System` implementation.
- `src/leaderboard/*` — this is latamboard publishing pipeline; split
  it out to a separate repo, don't drag it into new benchy.
- Most of `configs/` — the new schema is one small YAML per benchmark.
- `src/tasks/common/base.py|freeform.py|structured.py|multiple_choice.py|multimodal_*.py` — Handler god-object; the four modules replace it.
- `src/{prefect_compat,run_id_manager,signal_utils,task_completion_checker,outcome,logging_utils,config_manager,config_loader,gpu_config,generation_config}.py` — plumbing tied to the old pipeline; rebuild only if the new engine actually needs it.
- `.plans/*` — 16 legacy planning docs, unrelated to the vision.

## 8. Proposed layout

```
benchy/
├── VISION.md
├── README.md
├── pyproject.toml
├── src/benchy/
│   ├── task/                # Task, schema loaders, task registry
│   ├── scoring/             # BaseScorer, primitives, transforms, exports
│   │   ├── primitives/      # exact_match, wer, cer, f1, embedding_sim
│   │   ├── structural/      # field_wise, field_wise_weighted
│   │   └── transforms/      # binary, threshold, restrict, mean
│   ├── system/              # System protocol + implementations
│   │   ├── model.py         # raw HF / API models
│   │   ├── node.py          # model + prompt
│   │   ├── workflow.py      # composed
│   │   └── agent.py         # tool-using loops
│   ├── data/                # Data, loaders, cachers, validators
│   ├── benchmark.py         # composes the four peers
│   ├── loss.py              # `as_loss()` export + adapters (dspy, textgrad)
│   └── cli.py               # `benchy new|run|export-loss|list`
├── benchmarks/
│   └── <task>/<domain>/<language>/
│       ├── benchmark.yaml   # 4 fields: task/scoring/data/system
│       ├── scoring.py       # optional custom rubric
│       └── data/            # or a link out
├── tests/
└── docs/
```

Everything else in the current tree is either salvaged into these
locations or dropped.

## 9. Non-goals (for this spec)

- The Software-3.0 optimizer itself. We ship the `.as_loss()` hook and
  a DSPy adapter; the optimizer is a downstream project.
- A GUI. CLI + SDK only in this cut.
- Multi-tenant / hosted service. Local + endpoint-callable systems only.
- Migrating existing latamboard leaderboard publication logic — it's
  split out to its own repo.
- Backwards compatibility with current `configs/*.yaml`. New benchy is
  a new package; the migration is a rewrite per benchmark, not a
  schema translation.

## 10. How we get there

Cut the surface first, backfill later. The execution plan lives in
`docs/superpowers/plans/2026-06-22-new-benchy-execution.md`; here is
the shape:

1. Land the four module skeletons + `Benchmark` composition + a
   passing end-to-end for one reference benchmark (image extraction /
   invoices / es-AR). Nothing else.
2. Port the scoring primitives from `common/metrics.py` +
   `structured_metrics_calculator.py`.
3. Port the three custom adapters as `System` implementations.
4. Port the FLEURS transcription benchmark as second reference.
5. Nuke the legacy tree once (2) through (4) are green. This is the
   destructive step and requires user approval — nothing lands before
   that gate.
6. `as_loss()` + DSPy adapter as the last milestone in this spec's
   scope.

## 11. Open questions

- **Task schema language.** Pydantic vs raw JSON Schema vs both?
  Recommendation: pydantic as the authoring surface, JSON Schema as
  the exchange format. Decide before milestone 1.
- **System URL scheme.** `provider:model` vs `type://path` vs pluggable
  loaders? Recommendation: pluggable `System.register(scheme, loader)`
  with a small default set (`openai:`, `hf:`, `local:`, `endpoint:`).
- **Data validation strictness.** Fail loud on schema mismatch, or
  filter and warn? Recommendation: fail loud in `Benchmark.run()`,
  filter+warn in `Data.iter()`.

These are decided during milestone 1 with concrete code, not in this
spec.
