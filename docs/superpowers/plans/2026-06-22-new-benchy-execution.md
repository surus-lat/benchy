# New Benchy — Execution Plan

**Branch:** `REF/new-benchy`.
**Spec:** `docs/superpowers/specs/2026-06-22-new-benchy-design.md`.
**Salvage audit:** `docs/superpowers/specs/2026-06-22-new-benchy-salvage-audit.md`.
**Nothing destructive lands before user approval.** Milestone 5 is
the gate.

## Guiding rules

- **New tree lives at `src/benchy/`.** The old tree stays untouched
  under `src/` until the nuke milestone. This lets us build in
  parallel and cross-check.
- **Milestones are demo-shaped, not layer-shaped.** Every milestone
  ends with a runnable command. No purely-internal refactors.
- **Every port cites the source path.** From the salvage audit; keeps
  the trail auditable.
- **Nothing gets ported speculatively.** If a milestone doesn't need
  it, it doesn't move. We don't pay for our old plumbing twice.

## Milestone map

```
M1  Skeletons + reference benchmark plan   ← writes almost no code
M2  Scoring algebra                         ← primitives + transforms
M3  System abstraction                      ← model|node|workflow|agent
M4  Data + Task                             ← evidence + contract
M5  Benchmark composition + first run       ← end-to-end demo, gate 1
M6  Second reference (transcription)        ← proves the algebra reused
M7  as_loss() + DSPy adapter                ← optimizer hookup
M8  Nuke + rename (destructive)             ← user gate 2
M9  Docs, README, CONTRIBUTING              ← ship
```

## M1 — Skeletons + reference plan (day 1)

**Goal:** land the empty tree so we can commit against real paths in
later milestones.

Deliverables:
- `src/benchy/{task,scoring,system,data,benchmark,loss,cli}.py|/` — empty modules with docstrings and public API stubs.
- `benchmarks/image_extraction/invoices/es-AR/benchmark.yaml` — the
  four-field YAML for the reference benchmark, with placeholders.
- `tests/benchy/test_public_api.py` — one test that imports every
  public symbol. Guards the surface.

Ends when: `pytest tests/benchy/test_public_api.py` passes.

## M2 — Scoring algebra

**Goal:** the symbolic scorer works end-to-end for structured
extraction, in isolation from the rest of benchy.

Ports:
- `src/tasks/common/metrics.py` → `src/benchy/scoring/primitives/`
  (one file per primitive).
- `src/tasks/common/image_metrics.py` → `src/benchy/scoring/primitives/image.py`.
- `src/tasks/common/utils/structured_metrics_calculator.py` +
  `partial_matching.py` → `src/benchy/scoring/structural/field_wise.py`
  (default) + `field_wise_weighted.py` (weighted variant, sibling
  primitive per the earlier brainstorm rename).
- `src/tasks/common/utils/choice_utils.py` → primitives for MC scoring.

Writes:
- `src/benchy/scoring/base.py` — `Scorer` protocol with `evaluate`,
  `fitness`, `aggregate`, symbolic `repr`.
- `src/benchy/scoring/transforms/{binary,threshold,restrict,mean}.py`.
- `src/benchy/scoring/registry.py` — string→scorer lookup for YAML
  `family: ...` references.
- Tests: `test_field_wise_defaults.py`,
  `test_field_wise_weighted.py`, `test_binary_wraps.py`,
  `test_repr_round_trips.py`, `test_fitness_is_scalar.py`.

Ends when: `binary(field_wise(...))` scores a fake prediction end-to-
end without touching any other benchy module.

## M3 — System abstraction

**Goal:** a `System` implementation for the three shapes we already
know how to run.

Writes:
- `src/benchy/system/base.py` — `System` protocol, `SystemCapabilities`,
  `System.register(scheme, loader)`.
- `src/benchy/system/schemes/openai.py` — folds essence of
  `interfaces/openai_interface.py` + `openai_audio_interface.py`
  behind `System.load("openai:...")`.
- `src/benchy/system/schemes/hf.py` — folds
  `interfaces/transformers_audio_interface.py` + the three custom
  adapters (`voxtral_chat`, `qwen3_asr_chat`, `canary_nemo`) behind
  `System.load("hf:...")` with per-family routing.
- `src/benchy/system/schemes/endpoint.py` — generic HTTP endpoint
  (replaces the `http_interface` role for hosted systems).
- Tests: `test_system_load_openai.py`, `test_system_load_hf.py`,
  `test_system_capabilities.py`. Live-network tests marked
  `@pytest.mark.integration`.

Explicitly NOT ported: `generic_api_interface.py`, `http_interface`
scaffolding beyond what `endpoint.py` needs, `vllm_*`, `venv_manager`,
`probe/`.

Ends when: `System.load("openai:gpt-5-mini")` and
`System.load("hf:mistralai/Voxtral-Mini-4B-Realtime-2602")` both
return a working object that responds to a single-sample `run()`.

## M4 — Data + Task

**Goal:** load one dataset and validate it against one task schema.

Writes:
- `src/benchy/task/base.py` — `Task` object; input/output schemas
  authored as pydantic models (v2). JSON Schema is the exchange
  format.
- `src/benchy/task/registry.py` — resolves ontology paths
  (`task/domain/language`) to task objects.
- `src/benchy/data/base.py` — `Data` object with `iter()`, `schema
  validation`, `splits`.
- `src/benchy/data/sources/{hf,local,jsonl}.py` — porting the
  loading paths from `common/dataset_adapters.py` and
  `common/utils/dataset_utils.py`.
- `src/benchy/data/cache.py` — porting the cache logic from
  `common/dataset_loaders.py`.
- Tests: `test_task_schema_load.py`,
  `test_data_iter_validates.py`, `test_data_cache_hits.py`.

Ends when: the reference benchmark's Task and Data load, and
`Data.iter(schema=task.schema)` produces validated samples.

## M5 — Benchmark composition + first run (GATE 1)

**Goal:** end-to-end run of the reference benchmark on the new tree.

Writes:
- `src/benchy/benchmark.py` — `Benchmark(task, scoring, data,
  system)` with `async run(limit=None) -> RunResult`.
- `src/benchy/cli.py` — `benchy new`, `benchy list`, `benchy run
  <path> --system <url>` (minimum viable).
- `benchmarks/image_extraction/invoices/es-AR/benchmark.yaml` filled
  in for real.
- `benchmarks/image_extraction/invoices/es-AR/scoring.py` — the
  default rubric + a `strict = binary(default)` alternative.
- Data assets committed (or referenced) under
  `benchmarks/image_extraction/invoices/es-AR/data/`.

Demo: `benchy run image_extraction/invoices/es-AR --system openai:gpt-5-mini --limit 5` prints a scoring report.

**This is Gate 1.** User approval required before proceeding to nuke
anything in M8. If M5 doesn't demo cleanly, iterate here — do not
nuke.

## M6 — Second reference: FLEURS transcription

**Goal:** prove the four modules generalize by porting a
categorically different benchmark.

Writes:
- `benchmarks/transcription/fleurs/es-419/benchmark.yaml` + `pt-BR/`.
- Any missing scoring primitives (should be zero — WER/CER already
  land in M2).
- Any missing System scheme (should be zero — HF audio schemes land
  in M3).

Ends when: `benchy run transcription/fleurs/es-419 --system hf:whisper-large-v3-turbo` and `... --system hf:mistralai/Voxtral-Mini-4B-Realtime-2602` both produce valid results and match numbers from the current tree within tolerance.

## M7 — `as_loss()` + DSPy adapter

**Goal:** the vision's "new loss function" hook, end-to-end.

Writes:
- `src/benchy/loss.py` — `Benchmark.as_loss()` returns a callable
  `(system, examples) -> float`; also exposes the underlying scorer
  and task contract for optimizers that need structure.
- `src/benchy/loss/dspy.py` — thin adapter: `benchmark.as_dspy_metric()`
  returns a DSPy `Metric`. Exercised by a small optimizer smoke test.
- Docs example: optimize a node's prompt against
  `image_extraction/invoices/es-AR` using DSPy `BootstrapFewShot`.

Ends when: the DSPy smoke test converges (or provably runs one
iteration and updates a candidate), driven by the benchy loss.

## M8 — Nuke + rename (destructive, GATE 2)

**Goal:** collapse to a single tree.

Steps:
1. Confirm M5 and M6 demos still green on the new tree.
2. `git rm -r` the paths listed under **NUKE** in the salvage audit.
3. Move `src/benchy/` → `src/` if that's the preferred final layout
   (open question — could stay at `src/benchy/` for cleaner
   packaging).
4. Rewrite `pyproject.toml` `[project.scripts]` to point at the new
   CLI only.
5. Delete `.plans/*` (16 files).
6. Delete `configs/` except any files explicitly retained.
7. Regenerate `Makefile` targets against the new verbs.
8. Rewrite `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `CLAUDE.md`
   from scratch against the four modules.

**Gate 2:** each of the destructive steps above needs explicit user
approval. The default here is not to nuke a directory the user might
still want. Confirm per group.

Commit strategy: one commit per group so the nuke is reviewable and
revertable.

## M9 — Ship

- Merge `REF/new-benchy` → `main` after the user approves.
- Tag `v0.next` (or similar).
- Ship. The vision is now the implementation.

## Risk register

- **Pydantic-vs-JSON-Schema decision** (design open question) could
  ripple through M4. Decide by end of M1.
- **DSPy internals** may not accept our exact `Metric` shape; M7 might
  need a thin conformance layer. Not blocking earlier milestones.
- **HF adapter regressions** on M6: if numbers drift, the diff must be
  root-caused before nuking the old tree. This is why M8 waits.
- **Ontology mismatches** — some current benchmarks don't map cleanly
  to `/task/domain/language`. We handle by making all three levels
  optional (per the vision) and letting registry lookup fall back.

## What we are explicitly not doing

- Optimizer implementation itself (M7 ships the hook, not the
  optimizer).
- Latamboard publication logic (split repo).
- Backwards compatibility with the current `configs/*.yaml`.
- A GUI.
- A hosted service.

## Time envelope, best guess

- M1: 0.5d (skeletons)
- M2: 1–2d (mostly ports + tests)
- M3: 2–3d (real work; three schemes + custom family routing)
- M4: 1–2d
- M5: 1–2d (**Gate 1**)
- M6: 1d (proof of generality)
- M7: 1–2d (DSPy alignment)
- M8: 0.5d (destructive, gated)
- M9: 0.5d

Total: ~9–14 focused days.

## First actions after approval

1. Merge M1 immediately (empty skeletons + surface test).
2. Start M2 (scoring algebra) — safest early win, self-contained.
3. Everything after M2 assumes the algebra is stable.
