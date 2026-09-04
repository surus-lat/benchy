# The Declarative Spec Layer

The engine layer is behavior; a spec file is *names*. This layer compiles
one YAML file into the live object graph the engine already runs — through
the module registries only (`benchy.task.builtin`, the `benchy.scoring`
factories, `benchy.system.load`, `benchy.data.load`), the same surfaces the
seam tests pin. Zero engine changes, zero new semantics: a spec is a
call-tree; compiling it produces exactly the objects a hand-written
benchmark would build.

## One file, three semantic sections

    spec_version: 1
    exam:                       # fingerprinted — this IS the benchmark
      name: invoice-extraction-v1
      task:
        structured_extraction:
          ontology: structured_extraction/invoices/es-AR
          output: {type: object, properties: {vendor: {type: string},
                   total: {type: number}}, required: [vendor, total]}
      data:
        spec: jsonl:.data/invoices/train.jsonl
        expected: expected
      scoring:
        field_wise:
          fields: [vendor, total]
          per_field:
            exact_match: {case_insensitive: true}
    system:                     # the exam-taker — NOT fingerprinted
      url: openai:gpt-5-mini
      temperature: 0
    run:                        # operational defaults — NOT fingerprinted
      max_concurrency: 8

The separation is the semantics: comparability is a property of the exam.
Running the same `exam:` against N systems must not change the exam's
identity — so the fingerprint hashes `exam:` only. The GUI gets three
panels for the price of one file: an exam editor, a system picker, a run
configurator.

## The one resolution pattern

A node is one of:

- a scalar → itself
- a one-key mapping `{name: kwargs}` → registry lookup, `name(**kwargs)`

Nesting is **type-directed** and only inside `scoring`: a kwarg resolves
recursively iff its parameter is Scorer-typed (or its key names a
registered scorer factory). `task.*.output` and `data.input` are plain
data — never resolved. This is what makes scoring a compositional language
in YAML (`binary(invert(wer()))` is a tree, not a string).

Behavior is never serialized; behavior is named. When the registry can't
name it, the escape hatch is the same one production code uses — for
systems, `python:`. (v1 limit: tasks are the four builtins; a `python:`
task loader is v2 when a real exam needs it.)

## The compiler surface (`benchy/spec.py`)

- `compile_exam(doc_or_path) -> Benchmark` — task + data + scoring
- `compile_system(doc_or_path) -> System` — url + kwargs through `load`
- `fingerprint(doc_or_path) -> str` — sha256 over the canonical exam tree
  (task node, data node, scorer `repr` — which round-trips); formatting
  and comments do not affect it; system/run sections do not affect it
- `describe() -> dict` — the editable ontology as JSON: every task builtin,
  scoring factory (with params, annotations, defaults, first doc line),
  system scheme, data source. **This is the GUI contract**: the frontend
  renders forms from `describe()`; it never parses Python.

Errors are `benchy.core.SchemaViolation` (the spine error earns a second
consumer) and always list the known names — same policy as the system
registry.

## What this layer is for

1. `benchy eval --spec bench.yaml` — one command, whole benchmark (CLI
   wiring lands with the cycle-3 CLI unification; the two-CLI landmine).
2. The GUI frontend: edits the tree, the tree compiles, `describe()` is
   the form schema, `fingerprint()` is the change-detector ("you edited
   the exam; old scores are no longer comparable").
3. Exam portability: the file is the artifact someone shares; the engine
   fingerprint (git, version) travels in `run_outcome.json` as today.

## v1 limits (honest)

- Tasks: the four builtins only. Custom render/parse = write Python, v2.
- Runner: CLI flags stay authoritative; `run:` is defaults, not contract.
- Data paths: cwd-relative, no path magic against the YAML's location.
- `fingerprint` covers the spec tree, not the data *contents* (a dataset
  edit changes the exam but not the fingerprint until data v2 ships a
  content hash; noted, deliberately not built speculatively).

## Build order

1. `benchy/spec.py` (this commit) — lazy module imports so it imports
   everywhere, runs in composition.
2. Tests in `tests/benchy/integration/test_spec.py` — spec→objects→run
   end-to-end at the merge gate, fingerprint laws, describe contract,
   error surfaces.
3. CLI wiring + `run_outcome.json` fingerprint field — after the CLI
   unification fight.