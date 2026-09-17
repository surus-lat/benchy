# Benchy Engine 1.0 — Implementation Plan

Started 2026-09-16 21:53 -03. Branch `REF/benchy-v1.0`.

## Normative sources (priority order)

1. `paper/technical-paper-v10.3.md` — semantics + Appendix A engineering contract
2. `paper/benchy-engine-spec-v1.2.md` — normative MUST/MUST NOT conformance rules
3. `paper/benchy-engine-agent-handoff-v1.2.md` — implementation architecture
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
| `compiler.py` | strict YAML parse → validate → emit IR JSON | the single YAML→IR transformation (handoff §15 order) |
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
| `parser.py` | A 40-line `SafeLoader` subclass whose only consumer is the compiler. Lives in `compiler.py`. |
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
  benchy_v0/   previous benchy package (different architecture; NOT reused)
  tests_v0/    previous suite
src/           legacy, unused by the engine; its deps live in the `legacy` extra
```

## Phases

Each phase is TDD: tests first, then implementation, then the phase's exit criterion.
Gate command: `.venv/bin/python -m pytest tests -q`

- [x] **P0 — Scaffold.** Move old package/tests to `.attic/`. New `benchy/` skeleton,
      `errors.py`, pyproject repointed. Exit: `pytest tests -q` collects and passes empty.
- [x] **P1 — types.py.** Type vocabulary, schema compilation from YAML nodes, strict
      recursive value validation, leaf enumeration, exact-match equality.
      Exit: schema validation works with no YAML and no IR involved.
- [x] **P2 — compiler.py, parse half.** Strict YAML: duplicate keys, anchors/aliases/
      merge keys/custom tags rejected, exact top-level key set.
      Exit: C02, C03 pass.
- [x] **P3 — ontology.py.** Registry load by version, membership checks, the three task
      validators. Exit: C05–C10 pass (minus transcribe).
- [x] **P4 — scoring compile.** Leaf↔weight exact coverage, weight constraints,
      dimension emission with deterministic depth-first order. Exit: C11–C15, C28.
- [x] **P5 — compiler.py, whole.** Full validation order (handoff §15), IR emission.
      Exit: C01; same YAML → byte-identical IR.
- [x] **P6 — data.py.** JSONL streaming, row validation, workspace-confined artifact
      resolution, traversal/symlink rejection. Exit: C16–C19, C32.
- [x] **P7 — adapter.py + run.py.** Binding, invoke, status classification, result
      assembly. Exit: C20–C23.
- [x] **P8 — score.py + aggregation.** Weighted mean, contribution policy, benchmark
      mean. Exit: C24–C27, C29–C31.
- [x] **P9 — cli.py + acceptance test.** Exit: C33 — run from persisted IR with the
      source YAML deleted, byte-identical result.
- [x] **P10 — Full conformance sweep.** All C01–C33 green, ruff clean, LOC reported.

## Conformance matrix

C01–C33 from `benchy-engine-v1-agent-bundle/benchy-engine-build-plan-v1.md`, minus
C07/C08 (transcribe). Each C-case is a named test so the mapping is auditable:
`test_c01_canonical_extraction_benchmark_compiles`, etc.

## Phase 11 — provider adapters (designed, NOT built)

The build plan places this after conformance, which the engine has now passed. It is
the next real work, and it is what VISION means by "handling all the different
ai-system configurations under the hood".

**The leverage decision: build one adapter, not many.** An OpenAI-compatible
`/v1/chat/completions` endpoint covers OpenAI, vLLM, LM Studio, Ollama's compat mode,
Together/Groq/Fireworks, any self-hosted gateway, and a user's own agent behind a
route. One adapter parameterized by `base_url` reaches most of the space; a per-vendor
adapter for each reaches one each. Start with the one.

Where it goes: `benchy/providers/openai.py`. The core must not import it — verify with
a test that `benchy/*.py` contains no reference to `providers`.

What it owes the contract (`async invoke(dict) -> dict`), in order of difficulty:

1. **Output schema -> JSON schema.** Walk the output IR into a JSON-schema object for
   structured outputs. ~30 lines, recursive, mirrors `types.leaves`.
   `date`/`time`/`datetime`/artifacts become `string` with a format note in the prompt.
2. **Input object -> message.** Text fields inline; `image`/`audio`/`document` leaves
   are resolved absolute paths, so read and base64 them into the provider's content
   parts. The prompt comes from `ai-system.prompt` when present.
3. **Transport.** `urllib.request` keeps the engine's zero-dependency property intact;
   an `openai` SDK dependency would put a vendor package in the install path for
   everyone. Prefer stdlib unless something concrete forces otherwise.
4. **Response -> output object.** Parse the JSON body. Do **not** coerce types — the
   engine validates strictly, and a model returning `"121.00"` for a `float` *should*
   land as `invalid_output`. That is a true measurement of the system, not a defect to
   paper over.

Wiring: `--adapter` becomes optional, and `cli._run` falls back to
`providers.for_system(ir)` when `ai-system.type == model`. A missing provider stays
`adapter_not_bound` (A.11).

Testing without network or keys: run a `http.server` fixture on localhost that speaks
the chat-completions shape. That exercises every line above — schema generation,
encoding, transport, parsing — and is honest about what it does not prove (real
provider quirks).

Open question to settle first: `ai-system.parameters` is provider-specific by
definition (handoff §3). Decide whether the adapter passes it through verbatim or
validates a known subset. Verbatim is fewer parts and matches the spec's intent.

## Deferred, with reasons

- `transcribe` + field evaluators — Appendix E; needs the normalization decision
  (raw vs. specified NFKC+casefold+punctuation-strip) resolved first.
- Provider adapters — designed fresh (see Phase 11). Per the user's direction of
  2026-09-16, the old tree is a last-resort reference only, not a source to mine.
- Concurrency — spec permits sequential; add only on measured need.
- `translate` scores ≈0 for the same structural reason `transcribe` did. Implementing
  as specified; flagged once, not re-litigated.
