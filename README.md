# Benchy

A semantic language and execution engine for benchmarking AI programs.

Most benchmark frameworks are model-centric and exist to *run* benchmarks that
already exist. Benchy is neither. Its primitive is the **AI-system** — a model, a
model with an optimized prompt, a composed workflow, a tool-using agent, all the same
thing from outside — and its purpose is to help you *create* the benchmark that
represents your problem, because that benchmark is the first step in building the
system that solves it.

```
B = (P, S, D)      a benchmark is a program, a scoring function, and a dataset
R = (B, AI)        a run binds a benchmark to an AI-system
```

The benchmark is separate from the thing taking it.

## Install

```bash
pip install -e .
```

The engine's only dependency is PyYAML.

## A benchmark

One YAML file says what the program is, what "good" means, and where the exam lives.

```yaml
version: "1.0"
ontology_version: "1.0"

benchmark:
  task: extract          # /task/domain/language — the shared SURUS ontology
  domain: finance
  language: es

program:                 # the typed contract: P : X -> Y
  input:
    text: string
  output:
    invoice_number: string
    date: date
    supplier:
      name: string
      tax_id: string
    subtotal: float
    total: float

scoring:                 # the output leaves ARE the scoring dimensions
  weights:
    invoice_number: 1
    date: 1
    supplier:
      name: 1
      tax_id: 0          # validated, reported, but does not move the score
    subtotal: 1
    total: 5             # what the business actually cares about
  aggregator: weighted_mean

data:
  path: ./exam.jsonl

ai-system:
  type: external
  id: invoice-extractor-v7
```

The exam is JSONL, one `(input, expected)` pair per line:

```json
{"input": {"text": "Factura A-001 ..."}, "expected": {"invoice_number": "A-001", "...": "..."}}
```

## Run it

```bash
benchy run examples/invoices/benchmark.yaml \
  --adapter examples/invoices/system.py:extractor
```

```json
{
  "version": "1.0",
  "benchmark_score": 0.8148148148148149,
  "summary": {"examples": 3, "valid": 3, "invalid_outputs": 0, "execution_errors": 0},
  "results": [...]
}
```

`examples/invoices/` is real and runnable — the tests execute it.

## The AI-system

Benchy knows exactly one runtime contract:

```
named-field input object  ->  named-field output object
```

An adapter is anything that honours it. A function is enough:

```python
async def extractor(input_object):
    text = input_object["text"]
    ...
    return {"invoice_number": "A-001", "total": 121.0, ...}
```

Sync or async, a plain callable or an object with `invoke` — all accepted. Everything
about *how* your system runs (credentials, HTTP, an SDK, a local model, a whole agent
with a while-loop) lives on your side of that line. The engine has no provider
branches, and `type: model` is not a special execution path.

## Compile once, run from the IR

Valid YAML compiles deterministically into a canonical JSON IR. The engine consumes
only the IR and never reinterprets source, which you can check:

```bash
benchy compile benchmark.yaml -o ir.json
rm benchmark.yaml
benchy run ir.json --adapter system.py:extractor   # identical result
```

## What it refuses

There is no normalization layer and no repair. Compilation does not infer missing
semantics or inject defaults — it accepts or it rejects, with a structured diagnostic:

```json
{"phase": "compile", "code": "missing_weight", "path": ["supplier", "tax_id"],
 "message": "output leaf has no weight"}
```

Every output leaf needs exactly one explicit weight. Dataset rows and AI-system
outputs are validated strictly: a missing field, an extra field or a wrong type is
invalid. Duplicate YAML keys, anchors, aliases and merge keys are rejected, because
one semantic concept should have one syntax.

Three statuses, and the difference is load-bearing:

| status | meaning | score | contributes |
|---|---|---|---|
| `valid` | a schema-valid output, whatever it scored | `s_i` | `s_i` |
| `invalid_output` | returned something that is not a program output | `null` | `0` |
| `execution_error` | never produced an output at all | `null` | `0` |

A valid-but-completely-wrong answer scores 0 and is *not* the same as a failure.
Failures keep `null` so the diagnosis survives, and contribute 0 so they cannot
vanish from the average.

Dataset errors are neither — they abort the run. A malformed row says nothing about
the system under test, so scoring it would corrupt the measurement.

## Layout

```
benchy/
  errors.py      the one diagnostic shape
  types.py       semantic types: schemas, validation, equality
  ontology.py    /task/domain/language, and P ∈ P_T
  compiler.py    YAML -> canonical JSON IR
  data.py        the exam: streaming JSONL inside a workspace
  score.py       field correctness -> instance score -> benchmark score
  adapter.py     the runtime boundary
  run.py         the engine loop
  cli.py         compile / run
```

775 lines of code. There is exactly one representation of a schema anywhere in the
system — the IR JSON node — so nothing marshals between an internal form and the IR,
and nothing can drift.

## Scope

Ontology 1.0 has three tasks: `extract`, `classify`, `translate`. Field correctness is
exact match, and the instance aggregator is `weighted_mean`.

`transcribe` is deliberately absent. Exact match cannot rank transcription systems — a
transcript wrong by one word scores the same as one that is entirely wrong — so
declaring it fails compilation rather than producing scores that cannot be
interpreted. `audio` remains a valid input type. Paper Appendix E describes the
field-evaluator extension that would readmit it.

Also outside 1.0: variable-length output collections, custom evaluators, other
aggregators, adapter configuration in benchmark YAML, implicit weights. These are
extension points, not undefined behaviour.

## Documents

| | |
|---|---|
| `paper/technical-paper-v10.3.md` | semantics, plus the Appendix A engineering contract |
| `paper/benchy-engine-spec-v1.2.md` | normative conformance rules |
| `paper/benchy-engine-agent-handoff-v1.2.md` | implementation architecture |
| `VISION.md` | why benchy exists |
| `docs/engine-v1/` | the rebuild's plan and build log |

The paper, the spec, the handoff and the shipped registry are checked against each
other by `tests/test_doc_agreement.py`.

## Tests

```bash
python -m pytest tests -q
```

281 tests. Conformance cases C01–C33 from the build plan are named
`test_cNN_*`, and `tests/test_conformance_matrix.py` fails if any loses coverage.

`.attic/` holds the previous implementation, preserved in git history.
