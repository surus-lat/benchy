# New Benchy — Build Plan (fresh derivation)

**North star:** `VISION.md`. **Supersedes:** the execution plan in
`2026-06-22-new-benchy-execution.md` (kept for its salvage audit, which
stands). **Branch:** `REF/new-benchy`. **Package:** `benchy/` at repo root
(the old tree stays at `src/` until the gated nuke).

## What changed from the June spec, and why

### 1. The system is the argument, not a constructor field

June: `Benchmark(task, scoring, data, system)`.
Now: **`Benchmark = Task + Data + Scoring`**, and `benchmark.run(system)`.

The vision says a benchmark must export as "a new loss function" for
Software-3.0 optimizers. A loss function's free variable is the thing being
optimized — the AI-system. If the system is baked into the benchmark,
`as_loss()` is a bolt-on that has to reach in and swap a field. If the system
is the argument, then

```python
def as_loss(self) -> LossFn:
    async def loss(system): return (await self.run(system)).fitness
    return loss
```

is the whole implementation, and reusing one exam across many candidates —
the actual daily use of a benchmark — is the native path rather than a
rebuild. `System` stays a first-class peer *module* (the vision demands
four); it just isn't a field of the exam.

### 2. There is a Task↔System bridge, and the Task owns it

June's spec had `System.run(sample) -> Prediction`. That can't work: a
generic OpenAI system does not know how to turn an invoice image plus an
`InvoiceExtraction` schema into a request, nor how to coerce the answer back.
All the real engineering lives in that seam.

So the seam is explicit and belongs to the Task, which is the module that
owns the input/output contract:

```python
request    = task.render(sample, system.capabilities)   # Sample -> Request
response   = await system.invoke(request)               # opaque
prediction = task.parse(response, system.capabilities)  # Response -> typed
```

`Capabilities` is what makes "engineering must not be a problem" real: a
system with native structured output gets a schema-constrained request; one
without gets the schema in the prompt and repair on the way back. Same for
audio-in vs transcribe-then-prompt. The author never sees it.

No fifth module — `render`/`parse` are Task methods.

### 3. `Request` is transport-free

A `Request` is `messages: tuple[Message, ...]` of typed content parts
(`TextPart | ImagePart | AudioPart`) plus an optional `output_schema`. It
mentions no provider, no HTTP, no transformers. Each System lowers it into
its own transport. That is precisely the "hide all the different ai-system
configurations under the hood" clause of the vision, expressed as a type.

### 4. `benchy.core` is frozen first

Because five worktrees build in parallel, the contracts land *before* the
fan-out and nobody edits them. `benchy/core.py` (425 lines, stdlib-only,
imports no sibling) + `tests/benchy/test_core_contracts.py` (27 tests) is
the spine. Any disagreement between modules shows up as a failing spine test
at merge, not as a silent semantic drift.

## The shape

```
benchy/
  core.py          FROZEN. Ontology, Sample/Request/Response/Prediction,
                   Capabilities, the 4 protocols, Score/Record/Report,
                   LossFn, errors.
  task/            what shape does a solution have  (+ render/parse bridge)
  scoring/         what does "good" mean            (symbolic, composable)
  system/          the AI-program under test        (model|node|workflow|agent)
  data/            the evidence
  benchmark.py     Task + Data + Scoring -> run(system) -> Report
  loss.py          as_loss(), DSPy / TextGrad adapters
  report.py        rendering + persistence
  cli.py           new | list | run | export-loss
benchmarks/<task>/<domain>/<language>/benchmark.yaml
```

## Rounds

**Round 1 — five parallel worktrees, all against the frozen spine.**

| WT | Branch | Owns | Salvages from |
|----|--------|------|---------------|
| 1 | `wt/scoring` | `benchy/scoring/` — Scorer base, primitives, structural, transforms, registry, repr round-trip | `src/tasks/common/metrics.py`, `image_metrics.py`, `utils/structured_metrics_calculator.py`, `partial_matching.py`, `choice_utils.py`, `text_utils.py` |
| 2 | `wt/system` | `benchy/system/` — loader registry, `openai:`, `endpoint:`, `hf:`, `python:` schemes, capability detection | `src/interfaces/openai_interface.py`, `openai_audio_interface.py`, `transformers_audio_interface.py`, `src/adapters/*` |
| 3 | `wt/data` | `benchy/data/` — Data, sources (hf/jsonl/csv/glob/inline), cache, validation, splits | `src/tasks/common/dataset_adapters.py`, `dataset_loaders.py`, `utils/dataset_utils.py` |
| 4 | `wt/task` | `benchy/task/` — Task base, pydantic↔JSON Schema, render/parse + repair, built-in shapes, ontology registry | `src/tasks/common/task_config_schema.py` (intent only) |
| 5 | `wt/engine` | `benchy/{benchmark,loss,report,cli}.py` — run loop, concurrency, retry, Report, `as_loss()`, DSPy adapter, YAML, CLI | `src/engine/{retry,checkpoint}.py` (slimmed) |

Rules for every worktree: TDD, no edits to `benchy/core.py`, no imports of
sibling worktree modules (test against fakes), no `src/` imports in shipped
code (port, don't depend).

**Round 2 — integrate.** Merge all five, wire the reference benchmark
`image_extraction/invoices/es-AR` end to end, fix the seams the fakes hid.
This is the first real `benchy run`.

**Round 3 — generalize + optimize.** Second reference
`transcription/fleurs/pt-BR` on the salvaged ASR systems (proves the algebra
and the capability negotiation reuse), then `as_loss()` driving a DSPy
optimizer over a node's prompt.

**Round 4 — collapse (gated).** Delete the `src/` tree per the salvage
audit, rewrite README/AGENTS/CONTRIBUTING, ship. Destructive; needs
explicit approval.

## Decisions taken (June's open questions, closed)

- **Schema language.** Pydantic v2 is the authoring surface; JSON Schema is
  the exchange format carried on `Task.input_schema` / `output_schema` and
  on `Request.output_schema`. Core stays pydantic-free.
- **System URLs.** `scheme:rest`, pluggable via `System.register(scheme,
  loader)`. Ships with `openai:`, `endpoint:`, `hf:`, `python:`.
- **Validation strictness.** `Data.__iter__` filters and warns;
  `Benchmark.run` fails loud on schema violation.
