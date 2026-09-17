# Benchy Engine 1.0 — Implementation Plan

Started 2026-09-16 21:53 -03. Branch `REF/benchy-v1.0`.

## Normative sources (priority order)

1. `paper/technical-paper-v10.2.md` — semantics + Appendix A engineering contract
2. `paper/benchy-engine-spec-v1.1.md` — normative MUST/MUST NOT conformance rules
3. `paper/benchy-engine-agent-handoff-v1.1.md` — implementation architecture
4. `paper/v10-transcribe-removal-brief.md` — **APPLIED**: ontology 1.0 has three
   tasks (`extract`, `classify`, `translate`). No `transcribe`. No field evaluators.
5. `VISION.md` — why benchy exists; the four authoring concepts

## Optimization targets, in order

1. **Cleanest possible expression of the paper + VISION.** The code should read as
   the spec, not as a framework that happens to implement it.
2. **Fewest parts.** "The best part is no part." A module, class, protocol, or
   abstraction layer must be *forced* by the spec. Absent a forcing reason, delete it.
3. **Fewest lines.** Secondary to 1 and 2 — never compress at the cost of clarity.

## The one design decision that removes the most parts

**There is exactly one representation of a program schema in the entire system: the
IR JSON object.**

The handoff suggests internal `ObjectSchema` / `PrimitiveSchema` / `EnumSchema`
classes, plus an `ir` module that serializes them, plus a `results` module that
serializes result models. That is three parts and two marshalling layers to express
shapes the spec *already defines as JSON* (spec §16, §17).

So:

- the compiler emits IR schema nodes (`{"type": "object", "fields": {...}}`) directly
  from YAML in a single pass;
- `types.py` validates values, enumerates leaves, and tests equality by walking those
  same dicts;
- `run.py` emits result dicts matching spec §16 directly.

No dataclass↔JSON conversion exists anywhere, so it cannot drift, and the acceptance
test ("delete the YAML, rerun from persisted IR") is the natural path rather than a
special case.

## Module set: 8 files

Derived by taking the handoff's 10 proposed modules and deleting every one that isn't
forced.

| File | Owns | Forced by |
|---|---|---|
| `errors.py` | `BenchyError{phase, code, path, message}` + code vocabulary | spec §18; zero-dependency, imported by all |
| `types.py` | semantic type vocabulary, schema-node compilation, strict value validation, leaf-path enumeration, exact-match equality | A.1/A.2/A.6/A.7/§9 are per-type tables that must change together |
| `ontology.py` | versioned registry load, task/domain/language membership, the three task↔program validators | versioned external data shared across SURUS systems; adding ontology 1.1 must not touch the compiler |
| `compile.py` | strict YAML parse → validate → emit IR JSON | the single YAML→IR transformation (handoff §15 order) |
| `data.py` | JSONL streaming, per-example validation, workspace-confined artifact resolution | spec §7/§10; path-safety deserves isolation |
| `score.py` | field correctness → weighted mean → benchmark mean | VISION's scoring module; pure math, no I/O |
| `adapter.py` | the `invoke` protocol + id→adapter binding registry | *the* system boundary (A.10/A.11) |
| `run.py` | IR + adapter → result JSON | handoff §16 execution order |
| `cli.py` | `benchy compile` / `benchy run` | human surface |

### Parts deliberately NOT built

| Rejected part | Why |
|---|---|
| `ir.py` | IR is a JSON format, not code. Shape-checked once on load (~20 lines in `run.py`). |
| `results.py` | Results are a JSON format (spec §16). Emit dicts. |
| `parser.py` | A 40-line `SafeLoader` subclass whose only consumer is the compiler. Lives in `compile.py`. |
| `SchemaNode` class hierarchy | Superseded by the one-representation decision above. |
| field-evaluator seam (`wer`/`cer`) | Explicit non-goal. Appendix E describes it; v1.0 does not build it. |
| provider adapters | Outside engine core (A.11). Phase 11, after conformance. |
| async/concurrency machinery | Sequential is conformant. Concurrency is an optimization; only add if measured need. |

### Mapping to VISION's four authoring concepts

- **task** → `types.py` (what shape) + `ontology.py` (what operation)
- **scoring** → `score.py`
- **data** → `data.py`
- **ai-system** → `adapter.py`

The remaining four files are the compiler, the engine, the diagnostics, and the CLI.

## Layout

```
benchy/        NEW engine (the 8 modules)
tests/         NEW engine tests, mirroring the conformance matrix
.attic/
  benchy_v0/   previous benchy package (different architecture; mine for adapters)
  tests_v0/    previous suite
src/           legacy; source of provider adapters in a later phase
```

## Phases

Each phase is TDD: tests first, then implementation, then the phase's exit criterion.
Gate command: `.venv/bin/python -m pytest tests -q`

- [ ] **P0 — Scaffold.** Move old package/tests to `.attic/`. New `benchy/` skeleton,
      `errors.py`, pyproject repointed. Exit: `pytest tests -q` collects and passes empty.
- [ ] **P1 — types.py.** Type vocabulary, schema compilation from YAML nodes, strict
      recursive value validation, leaf enumeration, exact-match equality.
      Exit: schema validation works with no YAML and no IR involved.
- [ ] **P2 — compile.py, parse half.** Strict YAML: duplicate keys, anchors/aliases/
      merge keys/custom tags rejected, exact top-level key set.
      Exit: C02, C03 pass.
- [ ] **P3 — ontology.py.** Registry load by version, membership checks, the three task
      validators. Exit: C05–C10 pass (minus transcribe).
- [ ] **P4 — scoring compile.** Leaf↔weight exact coverage, weight constraints,
      dimension emission with deterministic depth-first order. Exit: C11–C15, C28.
- [ ] **P5 — compile.py, whole.** Full validation order (handoff §15), IR emission.
      Exit: C01; same YAML → byte-identical IR.
- [ ] **P6 — data.py.** JSONL streaming, row validation, workspace-confined artifact
      resolution, traversal/symlink rejection. Exit: C16–C19, C32.
- [ ] **P7 — adapter.py + run.py.** Binding, invoke, status classification, result
      assembly. Exit: C20–C23.
- [ ] **P8 — score.py + aggregation.** Weighted mean, contribution policy, benchmark
      mean. Exit: C24–C27, C29–C31.
- [ ] **P9 — cli.py + acceptance test.** Exit: C33 — run from persisted IR with the
      source YAML deleted, byte-identical result.
- [ ] **P10 — Full conformance sweep.** All C01–C33 green, ruff clean, LOC reported.

## Conformance matrix

C01–C33 from `benchy-engine-v1-agent-bundle/benchy-engine-build-plan-v1.md`, minus
C07/C08 (transcribe). Each C-case is a named test so the mapping is auditable:
`test_c01_canonical_extraction_benchmark_compiles`, etc.

## Deferred, with reasons

- `transcribe` + field evaluators — Appendix E; needs the normalization decision
  (raw vs. specified NFKC+casefold+punctuation-strip) resolved first.
- Provider adapters — mine `src/interfaces/` (openai 1005 LOC, google, surus) and
  `.attic/benchy_v0/system/hf/` after conformance.
- Concurrency — spec permits sequential; add only on measured need.
- `translate` scores ≈0 for the same structural reason `transcribe` did. Implementing
  as specified; flagged once, not re-litigated.
