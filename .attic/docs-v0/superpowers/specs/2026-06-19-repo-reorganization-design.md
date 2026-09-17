# Benchy Repo Reorganization — Design

**Status:** Design approved, ready for implementation plan.
**Date:** 2026-06-19
**Scope:** Internal reshuffle. Public surface (the `benchy` CLI, `run_outcome.json` schema, config formats, dataset spec) stays stable.

## Goals

1. Make the repo legible at a glance — someone landing on `main` can answer "where does X live?" in under a minute.
2. One canonical home for every file. Eliminate duplicate modules and half-finished migrations.
3. Keep the public surface stable: `benchy` CLI, config formats, JSON output schemas, dataset spec. All currently-importable paths keep working via shims for one release cycle.
4. Each migration step lands as its own commit, behind a green `pytest` + smoke run.

## Non-goals

- No rename of the top-level package (`src` stays `src`; not renamed to `benchy`).
- No change to the CLI command surface or any JSON output schema.
- No refactoring of business logic. Pure moves, deletes, and renames.
- No split into separate repos. `benchy-deck/` moves under `packages/` but stays in this repo.

## Top-level "after" tree

```
benchy/
├── README.md  CONTRIBUTING.md  CLAUDE.md  AGENTS.md
├── pyproject.toml  uv.lock  .python-version  .gitignore  env.example
│
├── src/                        # the package (Section 2)
├── tests/                      # mirrors src/ subpackages (Section 4)
├── configs/                    # unchanged shape, one rename (Section 4)
├── docs/                       # Diátaxis four-folder layout (Section 3)
├── scripts/                    # standalone CLI utilities
├── reference/                  # generated task/group manifests (stable contract path)
├── data/                       # committed dataset fixtures only
│
├── misc/                       # NEW — triage bucket for orphans
│
└── packages/
    └── benchy-deck/            # moved from repo root
```

## Section 1 — Janitorial sweep

### Locked deletes (build artifacts, no ambiguity)

- `__pycache__/` everywhere (root, `tests/__pycache__/`)
- `.DS_Store` everywhere
- `*.swp`, `*.swo` (root contains `.audit-ai-algorithms.md.swp` / `.swo`)
- `benchy.egg-info/` at root and `src/benchy.egg-info/`
- `.pytest_cache/`

### Moved to `misc/` (preserved, demoted for later triage)

- `deactivate/` → `misc/deactivate/`
- `eval.py` (376 bytes, root) → `misc/eval.py`
- `config_loader.py` (root shim that re-imports `src/config_loader.py`) → `misc/config_loader.py`
- `smolvlm_focus_expected.json` → `misc/smolvlm_focus_expected.json`
- `.agents/` (stale duplicate of `.agent/`) → `misc/.agents/` after merging any unique skill content into `.agent/skills/`

`misc/README.md` explains: *"Files demoted during the 2026-06-19 reorg because their owner / consumer wasn't obvious at move time. Before deleting, grep the codebase + your shell history for references. Anything still here in three months is safe to delete."* The dir is committed (visible triage), not gitignored.

### Moved with confidence

- `run_models.sh` → `scripts/run_models.sh`
- `public/benchy-diagram{1,2,3}.svg` → `docs/assets/`
- `benchy-deck/` → `packages/benchy-deck/`

### Flagged, not auto-touched

- `.env` at root: surfaced to the user during implementation. If committed and contains secrets, separate manual cleanup (history rewrite + gitignore) before any other work.
- `.notes/`, `.plans/`: local agent scratch, *not deleted*. Added to `.gitignore` so future commits don't pick them up.

### `.gitignore` updates

Add (idempotently, dedupe existing entries):
```
__pycache__/
*.py[cod]
*.sw[op]
.DS_Store
*.egg-info/
.pytest_cache/
.venv/
outputs/
logs/
.data/
.notes/
.plans/
.claude/worktrees/
```

`outputs/` and `logs/` get a tracked `.gitkeep` so the dirs survive but their contents don't. Currently-committed log files in `logs/` get moved to `misc/old-logs/` for safekeeping rather than deleted outright.

## Section 2 — `src/` internal layout

### Target tree

```
src/
├── __init__.py                 # back-compat re-exports
│
├── cli/                        # NEW
│   ├── __init__.py
│   ├── main.py                 # was src/benchy_cli.py
│   ├── eval.py                 # was src/benchy_cli_eval.py
│   └── probe.py                # was src/benchy_cli_probe.py
│
├── core/                       # NEW — cross-cutting runtime
│   ├── __init__.py
│   ├── config_loader.py
│   ├── config_manager.py
│   ├── generation_config.py
│   ├── gpu_config.py
│   ├── logging_utils.py
│   ├── outcome.py
│   ├── run_id_manager.py
│   ├── signal_utils.py
│   ├── task_completion_checker.py
│   └── prefect_compat.py
│
├── engine/                     # existing — gains pipeline.py
│   ├── benchmark_runner.py
│   ├── checkpoint.py
│   ├── connection.py
│   ├── output_diagnostics.py
│   ├── pipeline.py             # MOVED from src/pipeline.py
│   ├── protocols.py
│   └── retry.py
│
├── inference/                  # existing, unchanged
├── interfaces/                 # existing, unchanged
├── probe/                      # existing, unchanged
├── leaderboard/                # existing, unchanged
│
└── tasks/                      # existing, three cleanups (2a, 2b, 2c)
    ├── registry.py
    ├── group_runner.py
    ├── common/                 # one canonical home (2a)
    │   ├── base.py
    │   ├── freeform.py
    │   ├── multiple_choice.py
    │   ├── structured.py
    │   ├── multimodal_structured.py
    │   ├── multimodal_image_artifact.py
    │   ├── metrics.py
    │   ├── image_metrics.py
    │   ├── task_config_schema.py
    │   ├── config_generator.py
    │   ├── dataset_adapters.py
    │   ├── dataset_loaders.py
    │   ├── visualization.py
    │   └── utils/              # absorbs former src/common/
    │       ├── choice_utils.py
    │       ├── dataset_utils.py
    │       ├── field_diagnostics_report.py
    │       ├── partial_matching.py
    │       ├── schema_sanitizer.py     # moved from src/common/
    │       ├── structured_metrics_calculator.py
    │       ├── summary_reporter.py     # moved from src/common/
    │       └── text_utils.py
    ├── _templates/             # renamed from _template_handler/
    ├── classify/
    ├── document_extraction/
    ├── image_extraction/
    ├── image_manipulation/
    ├── portuguese/             # per-dataset folder migration finished (2b)
    ├── spanish/                # stays flat (2b)
    ├── structured_extraction/
    ├── transcription/
    └── translation/            # per-dataset folder migration finished (2b)
```

### 2a — `src/common/` deduplication

Today there are two homes for utility modules:
- `src/common/` — 4 files: `choice_utils.py`, `dataset_utils.py`, `schema_sanitizer.py`, `summary_reporter.py`
- `src/tasks/common/utils/` — 5 files including a second `choice_utils.py` and `dataset_utils.py`

**Action:** `src/common/` goes away. All four files move to `src/tasks/common/utils/`. Before each move:

1. Diff the duplicated file (`src/common/choice_utils.py` vs `src/tasks/common/utils/choice_utils.py`). If identical, drop the `src/common/` copy. If they diverge, merge the diff into the surviving file (or move the unique copy to `misc/` for review if the merge isn't obvious).
2. For files that exist only in `src/common/` (`schema_sanitizer.py`, `summary_reporter.py`), just move them into `src/tasks/common/utils/`.
3. Grep all callers and rewrite imports: `from src.common.X` → `from src.tasks.common.utils.X`.
4. Leave a one-line shim at `src/common/<name>.py` that re-exports from the new path.

### 2b — Task folder shape

Two patterns coexist:
- **Flat**: `tasks/<group>/<name>.py` — used by spanish, classify, structured_extraction, image_extraction, image_manipulation, transcription, document_extraction.
- **Per-dataset folder**: `tasks/<group>/datasets/<name>/task.py` — started for portuguese and translation only.

**Policy:** finish the per-dataset folder migration **only where it's already started** (portuguese, translation). Do not propagate the pattern to flat groups. Rationale: the per-dataset folder is heavier (deeper paths, more `__init__.py` files) and only earns its keep when the task needs colocated assets — download scripts, fixtures, dataset-specific metadata. Tasks that don't need those should stay flat.

Concretely:
- `tasks/portuguese/bluex.py`, `assin2_rte.py`, `assin2_sts.py`, `enem_challenge.py`, `oab_exams.py` → delete after diffing against `tasks/portuguese/datasets/<name>/task.py` (the newer copy).
- `tasks/portuguese/faquad_nli.py` — note: no flat file exists; only `datasets/faquad_nli/task.py`. Already migrated.
- `tasks/translation/flores.py`, `tasks/translation/opus.py` → delete after diffing against `tasks/translation/datasets/<name>/`.
- For each delete: if the flat file is *newer* than the folder version, that's the canonical copy — promote the flat content into the folder version, not the other way around. Use `git log` on both to decide.
- After all deletes, verify the task registry (`src/tasks/registry.py`) discovers the surviving tasks correctly via a smoke run.

### 2c — Back-compat shims

Every module that moves leaves a shim at its old path:

```python
# src/pipeline.py — DEPRECATED: moved to src/engine/pipeline.py
# Remove this shim after one release cycle (next minor version bump).
from src.engine.pipeline import *  # noqa: F401, F403
```

Shim policy:
- Module-level `*`-re-export. If `__all__` is missing, also re-export the public names explicitly so star-import works.
- Comment includes the new location and removal target.
- Shims are tracked in a one-off `MOVES.md` (also at `misc/MOVES.md`) so we can sweep them in a single follow-up commit.

Public-surface change: `pyproject.toml` `[project.scripts]` entry updates from `benchy = "src.benchy_cli:main"` to `benchy = "src.cli.main:main"`. This is the *single* externally visible change. The CLI command name and behavior are unchanged.

## Section 3 — `docs/` Diátaxis migration

Commit fully to the four-folder Diátaxis layout (tutorials / how-to / reference / explanation). Half-done today — the migration finishes here.

### Target tree

```
docs/
├── README.md                       # index — four-quadrant map + links
│
├── tutorials/
│   └── getting-started.md          # was tutorial-getting-started.md
│
├── how-to/
│   ├── evaluate-models.md          # was evaluating_models.md
│   ├── contribute-tasks.md         # was contribute_tasks.md
│   ├── contribute-providers.md     # was contributing_providers.md
│   ├── use-cli-datasets.md         # was CLI_DATASET_USAGE.md
│   └── run-tests.md                # was testing.md
│
├── reference/
│   ├── architecture.md             # was architecture.md (the "what" doc)
│   ├── cli.md                      # was reference-cli.md
│   ├── config.md                   # was reference-config.md
│   ├── tasks.md                    # was reference-tasks.md
│   ├── output-artifacts.md         # was reference-output-artifacts.md
│   ├── dataset-spec.md             # was DATASET_SPEC.md
│   ├── probe-contract.md           # was benchy_probe_contract.md
│   ├── scoring.md                  # was SCORING.md
│   ├── generation-config.md        # was GENERATION_CONFIG.md
│   └── handler-system.md           # was HANDLER_SYSTEM_GUIDE.md
│
├── explanation/
│   ├── architecture.md             # was explanation-architecture.md (the "why" doc)
│   └── vllm-version-management.md  # was VLLM_VERSION_MANAGEMENT.md
│
├── assets/
│   ├── benchy_2.png
│   └── benchy-diagram{1,2,3}.svg
│
└── superpowers/specs/              # design specs (this file lives here)
```

### Special-cases

- `docs/architecture.md` and `docs/explanation-architecture.md` both exist. They serve different purposes — keep both, sort into the right quadrant: the old "what" doc → `reference/architecture.md`; the newer "why" doc → `explanation/architecture.md`.
- `docs/benchy_2.png` keeps its filename (referenced in `README.md`) — rename later when the README image link is touched.

### Path rewrites

A single mapping file is generated up-front (old path → new path), then:
1. Every renamed file is `git mv`'d.
2. The mapping drives a `sed` sweep across `README.md`, `CONTRIBUTING.md`, every `*.md` in `docs/`, and every `*.py` that mentions a doc path (rare but check).
3. The sweep diff is reviewed before commit — no auto-apply across unrelated files.

### Convention going forward

Added to `docs/README.md`:
- File names: lowercase, dash-separated, no prefix (the folder names the quadrant).
- New doc? Pick a quadrant first. If you can't, it probably doesn't belong in `docs/`.

## Section 4 — Tests, configs, runtime dirs, agent dirs

### `tests/` mirror reshape

36 flat test files become subdirected to mirror `src/`. File names unchanged; only paths shift.

```
tests/
├── conftest.py
├── fixtures.py
├── cli/                  # test_benchy_cli_probe, test_cli_arguments
├── core/                 # test_config_loader, test_outcome, test_retry, test_protocols
├── engine/               # test_benchmark_runner, test_task_status_checkpoint, test_connection, test_probe_runner
├── interfaces/           # test_openai_audio_interface, test_transformers_audio_interface, test_surus_factura_interface
├── tasks/
│   ├── handlers/         # test_handlers_base, test_handlers_freeform, test_handlers_structured, test_handlers_multiple_choice, test_multimodal_structured
│   ├── common/           # test_dataset_loaders, test_dataset_adapters, test_dataset_utils, test_audio_preprocessing, test_image_preprocessing, test_config_generator, test_metrics_choice, test_metrics_extended, test_translation_metrics, test_wer_cer_metrics, test_field_diagnostics_report, test_structured_metrics_calculator, test_task_config_schema, test_pipeline_summary, test_registry_adhoc, test_group_runner
│   ├── transcription/    # test_transcription_handler, test_transcription_capabilities
│   ├── translation/      # test_translation_handler_metrics, test_fleurs_subtasks
│   └── document_extraction/  # test_facturas_argentinas_dataset_id
└── leaderboard/          # already exists, unchanged
```

`pytest` discovery (`tests/` root, `test_*.py` pattern) still finds everything. Verify with `pytest --collect-only` before committing the move.

### `configs/`

Single rename: `configs/tests/` → `configs/examples/`. The directory contains example configs (`image-extract-gpt4.yaml`, `latamboard-gptoss.yaml`, etc.), not test fixtures. Misleading name.

Top-level `configs/config.yaml` and `configs/models.txt` stay where they are.

### Runtime dirs (recap from Section 1)

| Dir | Status | Action |
|---|---|---|
| `data/` | committed dataset fixtures | keep + `README.md` describing purpose |
| `.data/` | runtime scratch | keep, ensure gitignored |
| `outputs/` | runtime outputs | gitignore, track `.gitkeep` |
| `logs/` | runtime logs (currently committed!) | move existing log files to `misc/old-logs/`, gitignore the dir |
| `reference/` | generated manifests | keep at repo root (stable contract path) |
| `public/` | SVG diagrams | merged into `docs/assets/` |

### Agent dirs

| Dir | Action |
|---|---|
| `.agent/skills/` | canonical, keep |
| `.agents/` | merge unique content into `.agent/skills/`, then move to `misc/.agents/` |
| `.claude/` | keep |
| `.notes/`, `.plans/` | gitignore (don't delete) |

## Migration sequence

Each step is its own commit. Each commit leaves `pytest` green and `benchy eval --limit 2` smoke-passing against a cheap config (e.g. `configs/examples/spanish-gptoss.yaml`).

1. **Janitorial pass** — deletes/moves of obvious junk + create `misc/`. Zero code changes.
2. **`.gitignore` update** — additions listed in Section 1. Verify `git status` is clean afterward.
3. **`docs/` Diátaxis migration** — `git mv`s + sed sweep of doc-path references in `README.md` / `CONTRIBUTING.md`.
4. **`tests/` mirror reshape** — pure `git mv`s. `pytest --collect-only` matches pre-move count.
5. **`src/cli/` extraction** — move three CLI files, add shims at old paths, update `pyproject.toml` `[project.scripts]`. Verify `benchy --help` and `benchy eval --help` still work.
6. **`src/core/` extraction** — 10 loose root files → `core/`, shims at old paths. Verify all tests still pass.
7. **`src/engine/pipeline.py` move** — same pattern, shim at `src/pipeline.py`.
8. **`src/common/` → `src/tasks/common/utils/` dedup** (Section 2a). Per-file diff first, merge into the survivor, delete the loser, shim.
9. **Task per-dataset folder finish** (Section 2b). Only portuguese + translation. Diff old flat file against new folder version, keep the canonical copy, delete the other.
10. **`configs/tests/` → `configs/examples/` rename** + grep for any references.

CI gate between each step:
- `pytest` clean
- `benchy eval --limit 2 --tasks <cheap-task> --exit-policy smoke` returns `0`
- `benchy datasets --json` and `benchy tasks --json` still respond

If any step fails the gate, revert and split — never bundle failures forward.

## Risks and rollback

- **Import-path drift in callers.** Mitigated by shims, but shims masquerade for one release. Anyone running off a forked `main` keeps working until they update; nobody on `main` breaks.
- **Hidden coupling to root-level paths.** Files like `eval.py`, `config_loader.py` (root) may be imported by something we can't see (scripts in user shell history, downstream agents). Mitigation: `misc/` retains them; not deleted.
- **Duplicate task files being out of sync.** For portuguese / translation, the flat-vs-folder duplicates may have drifted. Mitigation: explicit per-file diff before delete in step 9.
- **Pytest collection regression.** A `tests/` subdir without `__init__.py` could affect collection if conftest discovery quirks bite. Mitigation: `pytest --collect-only` count check.
- **Rollback.** Each step is a single commit. Revert one commit to undo one step. No step is destructive past `misc/` — actual deletes (the `misc/` content itself) come in a separate, later cleanup PR after a soak period.

## Open implementation-time questions (not blockers)

- Should `src/__init__.py` itself add `from . import cli, core, engine, ...` so `import src` exposes the subpackages? Or keep it empty?
- For Section 2b, when a flat task file is newer than the per-dataset folder version, do we *also* propagate the metadata.yaml / `__init__.py` shape, or just promote the Python content?
- For the `misc/MOVES.md` shim manifest — does it stay in `misc/` (visible) or move to `docs/superpowers/specs/` (alongside this design)?

These are decided during implementation; calling them out so they don't surprise us.
