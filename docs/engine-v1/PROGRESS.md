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

## 2026-09-17 09:45 -03 — clean-clone gate, and the bug it caught

Ran the clean-clone gate the old HANDOFF used: fresh `git clone`, fresh venv,
`pip install -e '.[dev]'`, full suite. It immediately caught a bug the 280-test suite
could not:

**`examples/invoices/exam.jsonl` was gitignored and absent from every clone.** A
repo-wide `*.jsonl` ignore — correct for datasets and run artifacts — also ate the
example's exam. The suite passed locally because the file existed locally; on a fresh
clone the README's headline command failed with `data_not_found`. Negated the ignore
under `examples/` and added a test asserting every example file is present and not
ignored, because the suite passing while the repo was broken is the exact failure mode
worth pinning.

Re-verified after the fix, from a clean clone at `dd4c425`: **281 passed**, and the
README's headline command produces the score the README quotes.

Lesson worth keeping: three separate defects this session were invisible to the test
suite and visible only to *running the real thing* — the missing `package-data` entry
for the ontology registry, the dependency list that pulled openai/pandas/datasets into
a supposedly PyYAML-only engine, and this. Tests check the code; only installing and
cloning check the artifact.

## 2026-09-17 09:20 -03 — heartbeat 2. Phase 11 BUILT.

Note on timestamps: the entries above drifted ahead of the real clock (I was
estimating). This one is the system clock. The 12-hour window ends 09:53, so ~33
minutes remain; suite green, all P0–P10 done, so the heartbeat's stop condition was
not met and the next real work was Phase 11.

Built it from the design note written last heartbeat — **`benchy/providers.py`, 149
code lines, 23 tests**, all against a local `http.server` speaking chat-completions,
so it needs no API key and no network.

Decisions, all as designed:
- **One adapter, not many.** OpenAI-compatible `/v1/chat/completions` reaches OpenAI,
  vLLM, LM Studio, Ollama, the hosted aggregators, any gateway. `OPENAI_BASE_URL`
  points it anywhere.
- **`urllib`, not the `openai` SDK.** Verified on a clean clone: runtime dependencies
  are still PyYAML alone. Installing benchy still installs no vendor package.
- **No type coercion.** A model returning `"121.00"` for a `float` yields
  `invalid_output`. Pinned by `test_the_adapter_does_not_coerce_types`. Repairing it
  would make the benchmark lie about the system under test.
- **A non-JSON reply is returned as raw text**, so the engine records it as the
  prediction and classifies it `invalid_output` — strictly more informative than
  raising, and it needs no special case anywhere.
- **Fail at setup, not per example.** Audio/document inputs, artifact outputs and a
  missing `OPENAI_API_KEY` raise at construction rather than producing a column of
  identical `execution_error`s.
- `parameters` passes through verbatim (the open question in the design note —
  verbatim is fewer parts and matches the spec's intent).
- CLI: `--adapter` is now optional and falls back to provider selection for
  `ai-system.type: model` (A.11). An explicit adapter still wins, asserted by a test
  that also checks the provider was never contacted.

The boundary holds: `test_the_engine_core_does_not_import_providers` asserts none of
the eight core modules references this file. Only `cli.py` does, which is A.11's
"the runtime may select a reusable provider adapter".

**The LOC guard added earlier today caught its first real drift**: `providers.py`
changed the count, and the README's figure was stale within the hour. Split into
"780 lines of code, plus 149 in the optional provider adapter", since `providers.py`
is deliberately outside the core and the README lists it separately.

Gate: **303 passed**, ruff clean, clean-clone verified, working tree clean.

## 2026-09-17 09:40 -03 — legacy cleanup, on the user's call

Four decisions taken by the user; all executed.

**Deleted outright** (git history preserves everything):
`src/` (169 tracked files), `configs/` (80), `.staging/` (57), `submissions/` (15),
`benchy-engine-v1-agent-bundle/` (superseded by `paper/`), `config_loader.py`,
`.agent/` (21 skills, 13 of which described the deleted architecture), `Makefile` and
`setup.sh` (tooling for the deleted tree), `scripts/` (ASR-panel and vLLM venv
management).

**Moved to `.attic/`** rather than deleted, being reference material:
34 stale `docs/*` files -> `.attic/docs-v0/`, the old `AGENTS.md`, and the
`publish-submission` workflow plus its PR template -> `.attic/workflows-v0/`.

**Rewritten because they were actively misleading:**
- `CLAUDE.md` — routed every future agent session into `.agent/skills/`, now deleted.
  Replaced with orientation, the gate, the core-import rule, the three optimization
  targets, and the two lessons this rebuild paid for.
- `AGENTS.md` — was "the machine-facing contract for running Benchy", describing the
  old CLI. Now a pointer to `CLAUDE.md`, kept only because tools look for the filename.
- `CONTRIBUTING.md` — its three "where to start" links pointed into `.attic/`.
- `pyproject.toml` — dropped the `src` package, the `benchy-legacy` script and every
  legacy extra (`legacy`, `local`, `prefect`, `document`, `translation`,
  `transcription`). Only `dev` survives. Version 0.1.0 -> 1.0.0, and the description
  no longer says "LATAMBoard benchmarking suite".

**Caught two things that would have broken silently:**
- `.github/workflows/ci.yml` ran `ruff check src tests`. With `src/` deleted, every
  push would have failed lint. Now `benchy tests`, and `pytest -q` -> `pytest tests -q`.
- `publish-submission.yml` triggered on `submissions/**` and ran
  `python -m src.leaderboard.merge_and_publish`. Both gone; workflow retired.

Verified: every `./path` and backticked repo path in README, CLAUDE, AGENTS and
CONTRIBUTING resolves. Tracked files 1100 -> 318. Suite **303 passed**, ruff clean,
CI steps simulated locally.

Heartbeat cron `8e09cc98` deleted — the 12-hour window is over.

**Left in place, for the user to call:** `misc/`, `proto/`, `reference/`, `logs/`,
`search/`, `.notes/`, `.plans/`, `.workshop/`, `.search/`, and the loose root files
(`ai-notes.md`, `IDEAS.md`, `HANDOFF.md`, `seed.md`, `canonical-json-ir.md.md`,
`Untitled.base`, `Untitled 1.base`, `uv.lock`, `env.example`). None of these are
referenced by the engine or its docs, but several look like personal notes rather than
project files, so they were not touched.

## 2026-09-17 — providers moved onto `llm-client`; Bedrock wired

User direction: solve Bedrock with SURUS's `llm-client` (github.com/surus-lat/llm-client).

**What `llm-client` is.** 11 files, ~1300 lines, httpx. `call()` with provider fallback,
a json_schema -> json_object -> none format ladder, 429 back-off, per-endpoint profiles
(OpenAI, Ollama, OpenRouter, Responses). **No Bedrock support at all** — zero mentions.

**Bedrock.** Checked AWS docs rather than memory: `bedrock-runtime.<region>/openai/v1`
accepts a Bedrock API key as a bearer token, which is exactly `llm-client`'s shape, so
Bedrock is one row in `_ENDPOINTS`. Confirmed live that the URL answers an invalid
token with an OpenAI-shaped 401. But Chat Completions on Bedrock does **not** serve
Claude (0/17), Nova (0/13) or Llama (0/12) — that needs a different request shape.

**Migrated `benchy/providers.py` onto `llm-client.call()`**, direct path only. benchy
keeps what is specific to benchmarking and refuses what would make a measurement lie:
no provider fallback (would score a mixture as one system), no format fallback (would
score an easier task), no injected defaults (`llm-client` defaults temperature 0.1,
max_tokens 2000), no silently dropped parameters, no coercion. Optional extra, imported
only when a model provider is selected.

**Live findings — four, none visible to the stand-in server:**
1. `python-httpx` is *not* on Together's block list (only `Python-urllib` is), so the
   User-Agent workaround goes away with urllib.
2. `llm-client` sends extra parameters as a literal nested `extra_body` key. Proven
   deterministic on Together: top-level `stop` truncates at "1, 2, 3, "; nested, it is
   ignored without an error. Hence benchy refuses such parameters at setup.
3. A null `max_tokens` is accepted, so benchy passes None and injects nothing.
4. **Truncation regressed.** With max_tokens 16 the provider reports finish_reason
   "length" and returns *partial JSON*. `llm-client` drops finish_reason, so benchy
   scored it `invalid_output` — "expected an object, got str". Scores are unaffected
   (null, zero contribution) but the diagnosis was lost. My stand-in test had passed
   because it scripted *empty* content: a stand-in only proves what you script into it.

**Upstream fixes prepared, not pushed.** Local branch `benchy/finish-reason-and-extra-body`
in a scratch clone of `llm-client`: expose finish_reason; merge extra_body into the body.
23 tests green. Proven live as a pair: max_tokens 16 goes from `invalid_output` to
"hit its token limit… raise max_tokens", and nested `stop` now truncates. benchy already
honours finish_reason when present (tested with an injected client), so it improves the
moment the fix lands. Not pushed because the second commit changes behaviour for the
internal backend.

Gates: clean clone `[dev]` — 286 passed, 42 provider tests skipped, installs benchy +
PyYAML only. Clean clone `[dev,providers]` — 328 passed, live example 3/3, score 1.0.

## 2026-09-17 — llm-client PR opened (user-authorised)

Pushed `benchy/finish-reason-and-extra-body` and opened
[surus-lat/llm-client#1](https://github.com/surus-lat/llm-client/pull/1), reviewers
`marianbasti` and `KennBro`, plus issues #2 and #3 for what it deliberately leaves open.

**The the internal backend question, answered before writing the PR.** `the-internal-backend` does not
depend on this package: every caller imports `src.services.llm_client`, its own vendored
module, and `llm-client` is absent from `backend/pyproject.toml`. So no production impact
today — the risk is at migration (`#108`/`#109`).

**That investigation changed the fix.** the internal backend's only `extra_body` use is
`{"enable_thinking": False}`, and its vendored client sends *both* the nested key and
`chat_template_kwargs.enable_thinking` — the latter being what vLLM reads. The package's
generic profile sent only the nested one. A plain flatten would have turned a
silently-ignored key into a top-level `enable_thinking` that a strict server may 400 on,
and neither form is the one vLLM honours. So commit 2 now maps `enable_thinking` to
`chat_template_kwargs`, matching the vendored behaviour exactly. The PR therefore
*removes* a migration landmine — had the internal backend migrated first, thinking suppression would have
silently stopped working and extraction would have begun failing to parse with nothing to
point at.

Lesson: the blast-radius check was worth more than the fix. Reading the consumer turned a
change that could have broken them into one that protects them.

## 2026-09-17 — Claude on Bedrock, live

User gave a Bedrock API key and asked for Claude. Region: "don't care unless important"
— it is. `us-east-1` carries 6 Claude models on mantle, `us-west-2` carries 1.

**Four dead ends, each checked rather than assumed**, before any code:

| route | result |
|---|---|
| `bedrock-runtime` `/openai/v1/chat/completions` | Claude absent (AWS table: 0 of 17 Anthropic models) |
| `bedrock-mantle` `/v1/chat/completions` | 400 — "does not support the '/v1/chat/completions' API" |
| `bedrock-mantle` `/anthropic/v1/messages` | path exists, but rejects Bedrock's `us.*` ids |
| `bedrock-runtime` `/model/{id}/converse` | **works** |

Plus the thing that made every early attempt fail: Anthropic models on Bedrock are
`INFERENCE_PROFILE`-only, so only `us.anthropic.…` / `global.anthropic.…` ids are
invocable. A bare id gets "on-demand throughput isn't supported", which reads like an
entitlement problem and is not one.

**Built `BedrockConverseProfile` in llm-client** (PR #4, stacked on #1). Converse differs
three ways and the profile absorbs all three: model in the URL path, `system` as a
top-level list, and no `response_format` — a schema becomes a *forced tool call* whose
`input` is returned as `content` JSON. That last choice is why **benchy needed no
Anthropic-specific code in `invoke` at all**: it still just `json.loads(content)`.

**A design bug my own test caught.** Routing was by hostname substring
(`"bedrock-runtime" in url`), so the local stand-in — and in production any
`BEDROCK_BASE_URL` pointing at a gateway — silently fell back to chat completions and
failed every example. The live run had only passed because the real hostname matched. Fixed
by letting a caller *state* the profile; benchy does, since it knows its own provider
table. Deterministic beats magic when a wrong guess means a failed run.

Live, all three backends, same benchmark, only `ai-system` differing:
Claude on Bedrock 3/3 · Together 3/3 · offline stand-in 0.8148.

And one honest measurement: Bedrock's `openai.gpt-oss-120b` **ignores `response_format`**
and emits `<reasoning>…` inline, so it scores 0. Confirmed by raw curl. benchy reporting
`invalid_output` rather than salvaging JSON from the prose is the no-coercion rule doing
its job.

Gates: 331 passed + 2 skipped against llm-client main (the two Converse tests skip via
`needs_converse`); 333 passed against the PR branch. ruff clean.
