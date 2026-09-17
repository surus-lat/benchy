# Benchy Engine 1.0 — Build Progress

Append-only. Newest entry at the bottom. This file is the source of truth for
"what is done / what is next" across session boundaries.

Gate: `.venv/bin/python -m pytest tests -q`

---

## 2026-09-16 21:55 -03 — session start

- Read and reconciled the normative set: v10.2 paper, spec v1.1, handoff v1.1,
  transcribe-removal brief, VISION.md.
- Established that v10.2 = v10 + implementation amendment; the transcribe-removal
  brief is **not yet applied to the paper** but **is applied to this implementation**
  (3 tasks: extract, classify, translate).
- Wrote `docs/engine-v1/PLAN.md`: 8 modules, the one-representation decision
  (IR JSON is the only schema representation — no dataclass marshalling layer),
  and phases P0–P10 with per-phase conformance-matrix exit criteria.
- Heartbeat cron `8e09cc98` scheduled every 3h at :47. NOTE: session-only — if the
  Claude session itself dies, the cron dies with it; this file is the durable state.

**Next:** P0 — scaffold.

## 2026-09-16 22:10 -03 — P0, P1, P2 done

- **P0 scaffold.** `git mv benchy .attic/benchy_v0`, `git mv tests .attic/tests_v0`
  (git recorded both as renames; nothing lost). New `benchy/` + `tests/`. pyproject
  console script repointed: `benchy = benchy.cli:main`, old one kept as
  `benchy-legacy`. `src/` untouched.
- **P1 `types.py`** (~250 LOC). Type vocabulary, `compile_schema`, `validate`,
  `leaves`, `at`, `equal`. Two design points worth keeping:
  - `validate` takes an optional `resolve` hook, so resolving dataset-relative
    artifact paths and validating the record are **one walk**, not two passes.
  - `_PARSE` (date/time/datetime) is shared by `validate` ("does it parse?") and
    `equal` ("do the parsed values match?"), so a temporal type is defined once.
- **P2 `compiler.py` parse half** (~95 LOC). Probed PyYAML first: `SafeLoader`
  already rejects custom and `!!python/...` tags, so `_StrictLoader` only adds
  anchors, aliases, merge keys and duplicate keys. Named the module `compiler.py`
  rather than `compile.py` to avoid shadowing the builtin.
- Gate: **100 passed**.

**Next:** P3 — `ontology.py`.

Note (user direction, 2026-09-16): do NOT lean on `.attic/benchy_v0` or `src/`.
Design fresh; consult the old tree only sporadically and only if genuinely needed.
The provider-adapter phase is now "design fresh", not "mine the old interfaces".

## 2026-09-16 22:35 -03 — P3, P4, P5, P6 done

- **P3 `ontology.py`** (~190 LOC) + `benchy/ontologies/1.0.yaml`. Registry resolved
  by *version token* (paths rejected outright), never by a filepath from the YAML.
  Refactored mid-phase into `check_classification` / `check_program` so handoff §15's
  order holds (membership before program grammar). The task table holds **one pair
  per task** — its language rule and its program rule — so a task is one edit point.
- **P4 `compile_scoring`** in `compiler.py`. Walks the weight tree *guided by the
  compiled output schema*, so leaf-vs-object is never guessed. Dimensions emit in
  output-schema order, not weight-mapping order.
- **P5 `compile_benchmark`**. Decision worth recording: **the compiler is a pure
  function `text -> IR`** — the only file it reads is the ontology registry.
  `data.path` is carried through verbatim (matching the example IR's
  `./data/invoices.jsonl`), so the IR is portable and the workspace is purely a
  runtime concept. Diagnostic order is pinned by four order tests.
- **P6 `data.py`** (~110 LOC). Generator, so rows stream; one containment check
  (`Path.resolve()` + `is_relative_to`) covers both traversal and symlink escape.
  The two path bases (workspace root for `data.path`, JSONL directory for artifacts)
  are separately tested.
- Gate: **214 passed**.

**Next:** P7/P8 — `adapter.py`, `score.py`, `run.py`.

Design decisions taken for the runtime layer, before writing it:
- `run(ir, workspace, adapter)` takes the adapter **directly** — no module-global
  registry in the execution path. A run evaluates exactly one AI-system, so
  lookup-by-id is the CLI's business, and `adapter.py`'s registry is off to the side.
- One normalizer, `invoker()`, accepts an Adapter instance or a plain callable,
  sync or async. Six lines instead of a class hierarchy.
- Dataset errors propagate from the generator *outside* the try blocks that classify
  adapter failures, so "abort the run" vs "score as execution_error" needs no flag.

## 2026-09-16 23:20 -03 — P7 through P10 done. ENGINE 1.0 COMPLETE.

- **P7 `adapter.py`** (37 code lines). `invoker()` normalizes an Adapter instance or
  a bare callable, sync or async, in six lines — which is why no adapter base class,
  wrapper class or `FunctionAdapter` exists. The `bind`/`resolve` registry is off the
  execution path; `run()` takes its adapter directly.
- **P8 `score.py`** (28 code lines) — the three levels of the paper as three
  functions, no I/O. `run.py` (69 code lines) — the loop, flat enough that the whole
  of Benchy's execution semantics is one screen.
- **P9 `cli.py`** + acceptance. `benchy compile` / `benchy run`. C33 verified two
  ways: as a test, and by hand through the installed entry point — compile, delete
  the YAML, rerun from `ir.json`, byte-identical result.
- **P10** ruff clean on the project config *and* on a broad sweep
  (F,E,W,I,UP,B,SIM,ARG,RET,C4,PIE). Six self-review cleanups applied, notably a
  branch in `types.equal` that was identical to its own fallback and a dead
  `except BenchyError: raise` in `parse`.
- Added `tests/test_conformance_matrix.py`: scans test names for `cNN` tokens and
  fails if any of C01–C33 (minus the withdrawn C07/C08) loses coverage.
- **Fixed a real packaging bug** found only by smoke-testing the installed CLI:
  `benchy/ontologies/1.0.yaml` is shipped data and had no `package-data` entry, so it
  would have been missing from a wheel.

### Final accounting

| | |
|---|---|
| modules | 8 + `__init__` + `cli` |
| code lines (no blanks/comments/docstrings) | **777** |
| file lines incl. docs | 1363 |
| tests | 268 passing, 1771 lines |
| dependencies | **stdlib + PyYAML** (verified by importing with every heavy dep blocked) |
| old tree, for comparison | 7801 lines across 49 files |

Definition of done (handoff §19): all 15 items met. Conformance matrix: 31/31
applicable cases green.

**Next:** the engine core is done, so the remaining work is, in order:
1. Apply `paper/v10-transcribe-removal-brief.md` to the paper (v10.2 -> v10.3) — the
   implementation is now *ahead* of the paper, the reverse of the brief's complaint.
2. `examples/` — a real runnable benchmark in-tree.
3. Phase 11 provider adapters, designed fresh (NOT mined from the old tree).
