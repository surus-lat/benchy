# Pipeline vocabulary mapping — benchy is the canonical task declaration

benchy's `benchmark.yaml` is the one canonical declaration of a task across the
pipeline repositories: datapipeline *projects* its `TaskContract` into it
(`src/datapipeline/benchy_projection.py` → `datapipeline.benchy_projection`),
and programpipeline *emits* it from its sealed contract
(`program_pipeline.benchmark_emit.emit_benchmark`) for `benchy_dev` to consume.
This table is the versioned, concept-by-concept mapping the three repos share;
the same factual table lives in `datapipeline/docs/BENCHY_MAPPING.md` and
`programpipeline/docs/benchy-mapping.md`.

| Concepto | benchy (canónico) | datapipeline | programpipeline |
|---|---|---|---|
| declaración de tarea | benchmark.yaml (program.input / program.output) | TaskContract (artifacts.py:51) | contract.yaml sellado (seal.build_contract) |
| campo de salida | hoja del árbol output (un peso por hoja) | io.output[] | expected |
| campo de entrada | hoja del árbol input | io.input[] | input |
| vocabulario cerrado | {enum: [values]} | type=enum + values | values por campo / labels |
| tipo de tarea | benchmark.task (ontología cerrada) | io.kind | task.type |
| metadata de campo | critical / derivation (NUEVO, inerte) | critical / derivation nativo | — |
| campo de entrada | text | texto | texto (heredado) |
| campo de salida | label | categoria | label |

## Field metadata: `critical` / `derivation` (new, inert)

A declared field may carry two optional metadata keys next to exactly one
structural form (`type` / `enum` / `fields`):

```yaml
program:
  output:
    total: {type: float, critical: true, derivation: derived}
```

- `critical`: boolean. The datapipeline meaning is conserved: a critical field
  must come out right in every row. benchy does not weight by it — scoring
  stays the paper's weighted mean over the declared weights.
- `derivation`: the closed vocabulary `copied` | `derived`. Where the value
  must come from: copied verbatim from the source system, or deduced from
  other fields. Absent = nothing declared (a model-produced value — what
  datapipeline calls *extracted* — is the exam's default semantics and carries
  no key).

The compiler **validates** the metadata and the IR **conserves** it verbatim
(compile_report). `validate`, `leaves`, `equal` and the scoring dimensions
never read it: metadata is reported, it never moves a score. Schemas without
metadata keys compile to the exact same IR as before. A field literally named
`critical` or `derivation` cannot be declared with the bare-mapping form — use
an explicit `fields:` object schema for such a parent.

## Rejected by design

- **`nullable` / `optional` / type unions**: rejected at the schema layer (paper
  A.10). datapipeline's `nullable` flag is a null-preparation concern and does
  not cross into the canonical declaration.
