# ROADMAP — three programs, one file interface

```
benchy-semantics   .yaml  → live objects + identity   [BUILT — 72/72]
benchy-engine      objects → Report/run_outcome        [BUILT — 431p]
benchy-agent       human request → .yaml              [NOT BUILT — next]
```

## The interface

The `.yaml` file is the contract between all three. The engine never learns
about YAML; the agent never learns about the engine. The GUI and the agent
are siblings — both *emit* `.yaml`, both are pure clients of the semantics
layer. One compiler, one engine, N frontends.

## What is already verified

- **semantics → engine mapping**: `compile_exam` produces exactly the objects
  the engine eats (`Benchmark`, `System`, run kwargs). Proven by the gate:
  YAML → compile → `run()` → fitness 1.0, and the pure-naming lock — the
  compiled exam has the same scorer repr and fitness as the handwritten one.
  Correctness is by construction: `describe()` and `compile_*` derive from
  the *same* registry functions. Single source of truth, cannot drift.
- **identity**: fingerprint laws locked (format/`system:`/`run:`/renames
  never change it; touching task/data/scoring always does).
- **the agent's vocabulary**: `describe()` — JSON-able, ~7KB, params/defaults/
  doc-lines/`scorer_typed`. This is the agent's (and GUI's) world model.

## What's missing, in dependency order

### S1 — `compile_run()` (~15 lines + tests)
The one real gap *inside* the mapping: the `run:` section validates but
does not compile. Map it to the engine's actual knobs — `limit`,
`concurrency` (`Benchmark.run` signature). Everything else in `run:`
(engine-internal retry/timeout policy) is deliberately not specable yet:
name it only when a consumer needs it (traffic, not topology).

### S2 — fingerprint → artifacts (small engine touch)
Stamp the exam fingerprint into `Report.meta` / `run_outcome.json`.
Comparability must be machine-checkable: same fingerprint + different
systems = comparable scores; different fingerprint = don't average.
Without this the fingerprint is a law without a police.

### S3 — `benchy run bench.yaml` (thin runner CLI)
Delivery. And it resolves the CLI fight by starvation: if `.yaml` is the
only input interface, the CLI degenerates to load → compile → run →
outcome. Both legacy CLIs (engine `cli.py` ontology-refs, wt/cli `eval.py`
flags) die — neither survives contact with a thinner runner.

### A1 — benchy-agent (the only net-new layer; zero engine code)
Contract: human request → `describe()` in the prompt context → write a
`.yaml` → `compile_exam()` is the validator (errors list known names — the
agent can self-correct) → `fingerprint()` confirms it built the exam it
intended → echo: smoke run before real spend.
The old `src/benchmark_compiler.py` is reference material only — it
injected args into the dead engine. The agent targets the registries.

### F1 — GUI (declared future, after A1 proves the contract)
`describe()` renders forms; `compile_*` is the save button; `fingerprint()`
is the dirty-check ("you edited the exam — old scores no longer compare").

### X1 — data-content hash v2 (filed, honest limit)
Today the fingerprint covers the spec tree, not the dataset bytes. The
upgrade: content-address the data (`sha256` of the materialized rows) and
fold it into the exam fingerprint. Not urgent — the spec *names* the
dataset; two datasets with the same name and different bytes is a
versioning problem before it is a hashing problem.

## Order rationale

S1–S3 complete the delivery path so a human can run what semantics
compiles. A1 and F1 are pure clients — they need nothing from the engine,
so they can be built in any order after S1 (A1 wants S2's stamped
fingerprint to prove smoke parity; F1 wants S3's runner to execute what it
saves). S3 is where the legacy CLIs die; do it before A1 so the agent has
one stable entrypoint to target.