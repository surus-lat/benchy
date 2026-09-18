# Benchy — Agent Guide

Benchy is a semantic language and execution engine for benchmarking AI programs.
Read `README.md` first; it is short and current.

## Orientation

The engine is `benchy/` — ten files, ~780 lines. Do not look for a framework; there
isn't one. The whole of the execution semantics fits in `run.py`.

| Where | What |
|---|---|
| `paper/technical-paper-v10.3.md` | semantics + the Appendix A engineering contract |
| `paper/benchy-engine-spec-v1.2.md` | normative MUST/MUST NOT conformance rules |
| `paper/benchy-engine-agent-handoff-v1.2.md` | implementation architecture |
| `VISION.md` | why benchy exists |
| `docs/engine-v1/PLAN.md` | the rebuild's design decisions and what was deliberately not built |
| `docs/engine-v1/PROGRESS.md` | append-only build log |

The paper, the spec, the handoff and the shipped ontology registry are checked
against each other by `tests/test_doc_agreement.py`. If you change the task
vocabulary in one, change it in all four or that test will tell you.

## Skills

`.agent/skills/<name>/SKILL.md`, invoked with the `Skill` tool or read directly.

| skill | when |
|---|---|
| `author-benchmark` | turning a business problem into a `benchmark.yaml` |
| `run-and-interpret` | running one, and reading the result correctly |
| `write-adapter` | exposing any AI-system through `invoke(dict) -> dict` |
| `add-provider` | adding a hosted provider, or spotting one that needs a new request shape |
| `extend-ontology` | adding a task — registry, validator, and the four documents that must agree |

## Working here

```bash
python -m pytest tests -q          # the gate, ~330 tests, ~12s
python -m ruff check benchy tests
```

Conformance cases from the build plan are named `test_cNN_*`;
`tests/test_conformance_matrix.py` fails if one loses coverage.

**The engine core is `errors, types, ontology, compiler, data, score, adapter, run`.**
It imports nothing beyond the standard library, PyYAML, and itself. `providers.py` is
outside that core and `cli.py` may select from it; nothing else may. Its transport is
SURUS's `llm-client`, an optional extra — `tests/test_acceptance.py` pins that neither
it nor `providers` leaks into the core.

## The standard this codebase is held to

Three rules, in order, from the rebuild:

1. **Cleanest expression of the paper.** The code should read as the spec, not as a
   framework that happens to implement it.
2. **Fewest parts.** "The best part is no part." A module, class, protocol or
   abstraction layer must be *forced* by the spec. Absent a forcing reason, delete it.
   `docs/engine-v1/PLAN.md` lists the parts that were rejected and why — read it
   before adding one.
3. **Fewest lines**, but never at the cost of clarity.

There is exactly one representation of a schema anywhere in the system: the IR JSON
node. No `SchemaNode` classes, no marshalling layer. Keep it that way.

## Two lessons worth inheriting

**Tests check the code; only installing and cloning check the artifact.** Four defects
in this rebuild were invisible to a green suite: a missing `package-data` entry for the
ontology registry, a dependency list that pulled vendor packages into a PyYAML-only
engine, an example dataset silently eaten by `.gitignore`, and a stale figure in the
README. Before claiming done, install into a fresh venv and run from a fresh clone.

**Prose drifts from code in silence.** That is the defect the transcribe-removal brief
was written to fix, and it had already recurred once. Numbers and vocabularies quoted
in documents are pinned by tests here; prefer adding a check to trusting a promise.

## Git commits

Never add `Co-Authored-By` trailers to commit messages.
