# Benchy Engine 1.0 — Build Plan and Conformance Matrix

## Goal

Implement the smallest engine that fully satisfies `benchy-engine-spec-v1.md`.

Do not begin by designing plugins, provider abstractions, generic HTTP DSLs, or dynamic scoring. First make the semantic compiler and one adapter path correct.

---

## Phase 1 — Semantic schema layer

Implement:

- schema AST;
- semantic type enum;
- enum schema;
- object schema;
- strict object validator;
- leaf enumeration.

Tests:

- primitive fields;
- nested fields;
- enum;
- empty input/output rejected;
- null rejected;
- list rejected;
- missing field;
- extra field;
- wrong type;
- nested path diagnostics.

Exit criterion: schema validation works independently of YAML.

---

## Phase 2 — Strict YAML parser

Implement:

- duplicate key detection;
- anchor/alias/tag rejection;
- top-level exact-key validation;
- source-location-aware diagnostics if practical.

Tests:

- valid canonical YAML;
- duplicate key;
- unknown root key;
- YAML alias;
- malformed YAML.

Exit criterion: raw source becomes one unambiguous parsed representation.

---

## Phase 3 — Ontology loading and classification

Implement:

- ontology YAML loader;
- version check;
- task/domain/language membership;
- task validator table.

Tests:

- unknown task/domain/language;
- classify valid/invalid;
- transcribe valid/invalid;
- translate valid/invalid language object;
- extract accepts general fixed named output schema.

Exit criterion: `P ∈ P_T` is enforced for ontology 1.0.

---

## Phase 4 — Scoring compiler

Implement:

- output leaf enumeration;
- mirrored weight-tree traversal;
- exact leaf coverage check;
- weight constraints;
- scoring-dimension IR generation.

Tests:

- flat weights;
- nested weights;
- missing weight;
- extra weight;
- intermediate-object weight;
- negative;
- NaN/infinity;
- all zero;
- deterministic path ordering.

Exit criterion: scoring IR is fully explicit.

---

## Phase 5 — Compiler and JSON IR

Implement compiler from parsed semantic source to typed IR.

The compiler:

- validates versions;
- resolves ontology semantics;
- compiles schemas;
- compiles scoring dimensions;
- validates data configuration;
- validates AI-system definition;
- emits IR.

Tests:

- same YAML → same IR;
- IR contains no unresolved source shorthand;
- engine can operate without source YAML.

Exit criterion: JSON IR is the sole engine input.

---

## Phase 6 — JSONL dataset loader

Implement streaming loader.

Tests:

- one row;
- multiple rows;
- blank lines;
- malformed JSON;
- missing input;
- extra root key;
- invalid input;
- invalid expected;
- empty dataset;
- missing artifact;
- relative artifact path resolution.

Exit criterion: dataset errors are distinct from AI-system errors.

---

## Phase 7 — Adapter protocol

Implement a test/in-process adapter first.

Example:

```python
class FunctionAdapter:
    def __init__(self, fn):
        self.fn = fn

    async def invoke(self, input_object):
        return await maybe_await(self.fn(input_object))
```

The exact helper API is implementation-specific; semantics are not.

Tests:

- valid output;
- adapter exception;
- returned extra field;
- returned missing field;
- wrong output type.

Exit criterion: runtime boundary is exercised without provider logic.

---

## Phase 8 — Artifact path runtime

Implement:

- dataset-relative resolution;
- absolute runtime path;
- regular-file/readability validation;
- byte-stream equality.

Use temp files in tests.

Tests:

- image/audio/document input path;
- path missing;
- artifact equality same bytes/different bytes;
- adapter receives resolved absolute path.

Exit criterion: no `Artifact` class is necessary.

---

## Phase 9 — Scoring runtime

Implement:

- exact-match evaluator;
- field-score generation;
- weighted mean;
- contribution policy;
- benchmark mean.

Tests:

- perfect score;
- partial score;
- zero-weight field;
- valid zero score;
- invalid output null/zero contribution;
- execution error null/zero contribution;
- failures stay in denominator.

Exit criterion: math matches spec exactly.

---

## Phase 10 — Results and diagnostics

Implement documented result JSON shape.

Tests:

- summary counts;
- field-score paths;
- error code/path;
- prediction retention;
- benchmark score;
- dataset failure returns no normal benchmark score.

Exit criterion: output is stable enough for UI/CLI consumption.

---

## Phase 11 — Built-in AI-system integrations

Only after the engine core passes conformance.

Implement provider/model adapters as separate integration code using the same adapter protocol.

Do not add provider branches to engine core.

---

# Conformance matrix

| ID | Case | Expected |
|---|---|---|
| C01 | canonical extraction benchmark | compile success |
| C02 | duplicate YAML key | compile error |
| C03 | unknown root key | compile error |
| C04 | variable-length schema | compile error |
| C05 | classify → enum output | compile success |
| C06 | classify → string output | task_program_mismatch |
| C07 | transcribe has audio + one string output | success |
| C08 | transcribe lacks audio | task_program_mismatch |
| C09 | translate single language scalar | error |
| C10 | translate source/target | success |
| C11 | missing leaf weight | compile error |
| C12 | extra leaf weight | compile error |
| C13 | zero weight among positive weights | success |
| C14 | all weights zero | compile error |
| C15 | negative weight | compile error |
| C16 | extra dataset input field | dataset error |
| C17 | missing expected field | dataset error |
| C18 | relative artifact exists | resolved absolute path |
| C19 | relative artifact missing | dataset error |
| C20 | adapter returns valid output | status valid |
| C21 | adapter returns wrong type | invalid_output |
| C22 | adapter returns extra field | invalid_output |
| C23 | adapter raises | execution_error |
| C24 | valid all-wrong output | status valid, score 0 |
| C25 | invalid output | score null, contribution 0 |
| C26 | execution error | score null, contribution 0 |
| C27 | one failure among two perfect examples | benchmark score 0.5 |
| C28 | nested output | field paths arrays |
| C29 | exact date semantic equality | correct |
| C30 | same artifact bytes | correct |
| C31 | different artifact bytes | incorrect |
| C32 | empty dataset | run data error |
| C33 | engine run from IR with YAML unavailable | success |

---

# Recommended implementation order

```text
schema
→ strict parser
→ ontology
→ scoring compiler
→ IR/compiler
→ JSONL loader
→ adapter protocol
→ runtime scoring
→ results
→ provider integrations
```

Do not start from provider integrations.

---

# Final acceptance test

Build one end-to-end fixture:

```text
benchmark YAML
+
ontology YAML
+
2-row invoice JSONL
+
fake adapter
```

Run:

```text
parse
→ validate
→ compile IR
→ execute
→ result JSON
```

Then delete/unavailable the source YAML and rerun execution from persisted IR.

The second run should behave identically, proving that the engine depends on compiled semantics rather than reinterpreting source.
