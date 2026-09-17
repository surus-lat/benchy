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

## 2026-09-17 08:45 -03 — heartbeat 1. Paper sync done (queued item 1).

Resumed from green: 275 passed, ruff clean, nothing uncommitted.

- Applied `paper/v10-transcribe-removal-brief.md`, producing **`technical-paper-v10.3.md`**.
  All five edits landed; all of the brief's §4 consistency checks pass:
  `transcrib` appears only in §10 and Appendix E, Appendix B and A.4 list the same
  three tasks in the same order, `audio` survives in 11 places, §9's canonical YAML
  still uses `extract`, and no "four tasks" prose exists.
- Wrote **Appendix E** (field evaluators as a described extension, not 1.0 behavior).
  Preserved the brief's technical claims and `evaluators:` syntax sketch. Added two
  things the brief left implicit and a future implementer needs:
  - keeping the raw numerator/denominator, not just the ratio, because mean-of-
    per-example is a macro-average while published WER is a micro-average;
  - the measured size of the normalization question — 9 to 15 points of per-example
    word accuracy across the ASR predictions stored in this repo. The brief said the
    decision "deserves its own treatment"; now it has a number attached.
- Synced the other two normative docs, which still contradicted the engine:
  **`benchy-engine-spec-v1.2.md`** and **`benchy-engine-agent-handoff-v1.2.md`**.
- Added `tests/test_doc_agreement.py` (7 tests): the paper's Appendix B and A.4, the
  spec's §5 and the handoff's validator table must all equal the shipped registry.
  **Verified the guard fails when it should** by smuggling `transcribe` back into
  `ontologies/1.0.yaml` — 4 of 7 went red, then restored.
  Skips cleanly when `paper/` is absent, so an installed distribution is unaffected.

Gate: **275 passed**, ruff clean on both rulesets.

**Next:** 2. `examples/` in-tree runnable benchmark. 3. Phase 11 provider adapters,
designed fresh.

## 2026-09-17 09:10 -03 — examples, README, dependency cleanup

- **`examples/invoices/`** — the paper's canonical extraction benchmark, runnable
  offline. Exercises nested output, mixed semantic types and explicit weights: the
  stand-in system gets five of six fields right on the third example but misses the
  one carrying weight 5 of 9, so it scores 4/9 and the benchmark scores 22/27.
  `supplier.tax_id` has weight 0, so the score is identical whether the system
  extracts it correctly or not — which is what a zero weight *means*.
- **`tests/test_examples.py`** — every `examples/*/benchmark.yaml` compiles and runs
  through the real CLI, and the invoices example is pinned to the exact score the
  README quotes. Documentation that is never executed rots.
- **Rewrote `README.md`.** The old one (31,707 bytes) described the architecture now
  in `.attic/`; moved to `.attic/README-v0.md`. Verified every claim in the new one
  against a real run — which caught the quoted score's last digit (`...149`, not
  `...148`) and confirmed the `missing_weight` diagnostic reproduces verbatim.
- **Fixed a dependency lie the README rewrite exposed.** `pyproject` listed openai,
  anthropic, pandas, datasets, scipy, pillow and more as *core* dependencies — all of
  them requirements of the legacy `src/` tree, none of them used by the engine. Moved
  to a `legacy` extra; dropped `jiwer` from `dev` (no evaluators in 1.0). Core is now
  `pyyaml` alone. Verified by building a fresh venv, `pip install -e .`, and running
  the example end-to-end: the venv contains PyYAML and benchy, nothing else.

Gate: **279 passed**, ruff clean, lean install verified.

**Next (not started — deliberately):** Phase 11 provider adapters. See the design note
below; starting a ~150-line integration with under an hour left would have left a
half-built part, which is worse than none.

## 2026-09-17 09:32 -03 — close-out of the 12-hour window

- Marked P0–P10 done in `PLAN.md` and fixed its own stale references: `compile.py` ->
  `compiler.py` (renamed during P2), normative sources -> v10.3/v1.2, and the two
  lines that still described the old tree as a source to mine for adapters.
  Verified every path `PLAN.md` names resolves; the one that does not is
  `benchy/providers/openai.py`, which Phase 11 explicitly has not built.
- Added **Phase 11 as an executable design note** rather than a half-built module.
  The leverage decision is recorded: one OpenAI-compatible adapter parameterized by
  `base_url` reaches OpenAI, vLLM, LM Studio, Ollama, the hosted aggregators and any
  self-hosted gateway; per-vendor adapters reach one each. Also recorded: use
  `urllib` rather than the `openai` SDK so the zero-dependency property survives, and
  do **not** coerce types in the adapter — a model returning `"121.00"` for a `float`
  *should* land as `invalid_output`, because that is a true measurement.
- Caught and fixed my own drift: the README claimed 777 engine lines, but the P10
  review cleanups made it 775. Added a test that recomputes the figure from source,
  so that number cannot go stale again.

Final gate: **280 passed**, ruff clean on both rulesets, working tree clean,
five commits on `REF/benchy-v1.0`.

### Where things stand

Engine 1.0 is complete and conformant. Definition of done (handoff §19): 15/15.
Conformance matrix: 31/31 applicable cases, each a named `test_cNN_*`.

Not done, deliberately, each with a reason recorded above:
- **Phase 11 provider adapters** — designed, not built. Under an hour left in the
  window; a half-built integration is worse than an executable design note.
- **Concurrency** — sequential is conformant; spec §16 makes it an optimization.
  Add on measured need, not on principle.
- **Field evaluators / `transcribe`** — explicit non-goal. Paper Appendix E holds the
  design, including the measured 9–15 point size of the normalization decision.
