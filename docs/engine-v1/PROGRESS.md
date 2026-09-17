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
