# Benchy Repo Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the benchy repo so every file lives in a named, role-described location, without changing the public surface (`benchy` CLI, configs, `run_outcome.json` schema, dataset spec).

**Architecture:** Each task lands as its own commit, gated by `pytest` staying green and a `benchy eval --limit 2 --exit-policy smoke` run succeeding. Moved modules leave one-line back-compat shims at their old paths for one release cycle. Ambiguous orphan files go to a `misc/` triage bucket rather than being deleted.

**Tech Stack:** Python 3.12, pytest, `uv` (or `pip -e .`), Diátaxis docs convention.

## Global Constraints

- Public surface is frozen: `benchy` CLI flags, `run_outcome.json` schema, `task_status.json` schema, `configs/` format, `docs/DATASET_SPEC.md` semantics — none of these change.
- Every moved Python module leaves a shim at its old path that re-exports the new one. Shim format:
  ```python
  # <OLD_PATH> — DEPRECATED: moved to <NEW_PATH>.
  # Remove this shim after one release cycle.
  from <NEW_PATH> import *  # noqa: F401, F403
  ```
- Each task ends with `pytest -q` clean and `benchy eval --tasks spanish.copa_es --provider openai --model-name gpt-4o-mini --limit 2 --run-id reorg_smoke_<task_number> --exit-policy smoke` returning exit code `0`. If you don't have an OpenAI key in the env, substitute the cheapest config you do have access to; record the substitution in the commit message.
- No commit may bundle two tasks. Every commit is one task and is independently revertable.
- Commit prefix: use `chore:` for moves and renames, `refactor:` for moves that include import rewrites, `docs:` for `docs/` work. Never add `Co-Authored-By` trailers (project policy in `CLAUDE.md`).
- Run all commands from repo root (`/Users/dobleefe/benchy`) unless stated otherwise.

---

## File Map (what gets touched, where it lands)

**Created directories:**
- `misc/` (triage bucket; committed; has README)
- `misc/old-logs/` (for currently-committed log files)
- `packages/` (for `benchy-deck/`)
- `docs/tutorials/`, `docs/how-to/`, `docs/reference/`, `docs/explanation/`, `docs/assets/`
- `src/cli/`, `src/core/`
- `tests/cli/`, `tests/core/`, `tests/engine/`, `tests/interfaces/`, `tests/tasks/handlers/`, `tests/tasks/common/`, `tests/tasks/transcription/`, `tests/tasks/translation/`, `tests/tasks/document_extraction/`

**Deleted (build artifacts only):**
- `__pycache__/` (all locations)
- `.DS_Store` (all locations)
- `*.swp`, `*.swo` (root has two)
- `benchy.egg-info/` (root + `src/`)
- `.pytest_cache/`

**Moved into `misc/`:**
- `deactivate/`, `eval.py`, `config_loader.py` (root shim), `smolvlm_focus_expected.json`, `.agents/`

**Moved with new home:**
- `run_models.sh` → `scripts/`
- `public/*.svg` → `docs/assets/`
- `benchy-deck/` → `packages/benchy-deck/`
- `src/benchy_cli*.py` (3 files) → `src/cli/`
- `src/pipeline.py` + 9 other loose root-of-src modules → `src/engine/` (pipeline only) and `src/core/` (the other 9)
- `src/common/*.py` → `src/tasks/common/utils/`
- Flat duplicate task files (`tasks/portuguese/{bluex,assin2_rte,assin2_sts,enem_challenge,oab_exams}.py`, `tasks/translation/{flores,opus}.py`) → deleted after diff confirms canonical copy exists in `datasets/<name>/task.py`
- `configs/tests/` → `configs/examples/`
- Currently-committed `logs/*` → `misc/old-logs/`

**Docs renamed/moved:** see Task 3.

**Tests reshaped:** see Task 4.

**Single `pyproject.toml` edit:** `[project.scripts]` `benchy = "src.benchy_cli:main"` → `benchy = "src.cli.main:main"` (in Task 5).

---

## Task 1: Janitorial sweep + `misc/` triage bucket

**Files:**
- Create: `misc/README.md`, `misc/old-logs/.gitkeep`
- Delete: build artifacts listed below
- Move: ambiguous orphans into `misc/`, plus three confident moves

**Interfaces:**
- Consumes: nothing.
- Produces: clean repo root + populated `misc/` bucket. Subsequent tasks rely on `misc/` existing.

- [ ] **Step 1: Baseline check — record current test count**

```bash
pytest --collect-only -q 2>&1 | tail -5
```

Expected: prints a count like `N tests collected in Xs` (no errors). Note `N` — every subsequent task must collect the same count (until Task 8 deletes any tests, which there shouldn't be).

- [ ] **Step 2: Delete safe build artifacts**

```bash
find . -type d -name '__pycache__' -not -path './.git/*' -not -path './.venv/*' -exec rm -rf {} +
find . -type f -name '.DS_Store' -not -path './.git/*' -exec rm -f {} +
find . -type f \( -name '*.swp' -o -name '*.swo' \) -not -path './.git/*' -exec rm -f {} +
rm -rf benchy.egg-info src/benchy.egg-info .pytest_cache
```

Run `git status` afterward. Confirm everything removed was either untracked (no entry) or shown as deleted. If anything tracked was deleted that looks load-bearing, stop and ask.

- [ ] **Step 3: Create `misc/` with README**

```bash
mkdir -p misc/old-logs
touch misc/old-logs/.gitkeep
```

Create `misc/README.md` with this content:

```markdown
# misc/ — Triage bucket

Files moved here during the 2026-06-19 repo reorganization because their
owner or consumer wasn't obvious at move time. Before deleting anything
from this directory:

1. `grep -r <filename>` across the repo — confirm nothing imports or
   references it.
2. Check your shell history and any external scripts you run.
3. If still unclear after 90 days, it is safe to delete.

See `docs/superpowers/specs/2026-06-19-repo-reorganization-design.md`.
```

- [ ] **Step 4: Move ambiguous orphans into `misc/`**

```bash
git mv deactivate misc/deactivate
git mv eval.py misc/eval.py
git mv config_loader.py misc/config_loader.py  # the root shim, not src/config_loader.py
git mv smolvlm_focus_expected.json misc/smolvlm_focus_expected.json
```

For `.agents/` (note the trailing 's', distinct from `.agent/`):

```bash
# First, copy any unique skills into .agent/skills/ (currently .agents/ only contains add-task/.DS_Store, so likely nothing to merge)
ls .agents/skills/ 2>/dev/null
# Confirm only add-task/ is present and contains nothing unique. Then:
git mv .agents misc/.agents
```

- [ ] **Step 5: Move confident-destination files**

```bash
git mv run_models.sh scripts/run_models.sh
mkdir -p docs/assets
git mv public/benchy-diagram1.svg docs/assets/
git mv public/benchy-diagram2.svg docs/assets/
git mv public/benchy-diagram3.svg docs/assets/
rmdir public
mkdir -p packages
git mv benchy-deck packages/benchy-deck
```

- [ ] **Step 6: Move committed log files into `misc/old-logs/`**

```bash
# Identify currently-tracked log files
git ls-files logs/
# Move them
git mv logs/*.log misc/old-logs/ 2>/dev/null || true
# If subdirs exist under logs/ and contain tracked files, move them too:
for d in logs/*/; do
  if [ -n "$(git ls-files "$d")" ]; then
    git mv "$d" "misc/old-logs/$(basename $d)"
  fi
done
# Keep logs/ as a dir for runtime use
mkdir -p logs
touch logs/.gitkeep
git add logs/.gitkeep
```

- [ ] **Step 7: Verify baseline still holds**

```bash
pytest --collect-only -q 2>&1 | tail -5
```

Expected: same `N tests collected` as Step 1.

- [ ] **Step 8: Flag `.env` to the user (do not auto-delete)**

```bash
ls -la .env 2>/dev/null && echo "WARNING: .env exists at repo root — verify it does not contain secrets before next step. If it does, stop and surface to the user."
```

If `.env` contains any keys, do not proceed. Surface to the user.

- [ ] **Step 9: Commit**

```bash
git add -A misc/ scripts/run_models.sh docs/assets/ packages/benchy-deck/ logs/.gitkeep
git status  # confirm staged set
git commit -m "chore: janitorial sweep, misc/ triage bucket, demoted orphans

Deletes build artifacts (__pycache__, .DS_Store, *.sw[op], egg-info,
.pytest_cache). Creates misc/ for ambiguous orphans (deactivate/,
eval.py, root config_loader.py shim, smolvlm fixture, stale .agents/).
Moves run_models.sh to scripts/, public SVGs to docs/assets/,
benchy-deck to packages/, currently-committed logs to misc/old-logs/."
```

---

## Task 2: `.gitignore` update

**Files:**
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Task 1's deletions (so the patterns being added match real removed paths).
- Produces: a `.gitignore` that prevents the deletions in Task 1 from coming back.

- [ ] **Step 1: Read current `.gitignore`**

```bash
cat .gitignore
```

Note what is already covered so the additions in Step 2 don't duplicate.

- [ ] **Step 2: Add missing patterns**

Open `.gitignore` and ensure each of these patterns exists exactly once. Add what is missing (do not remove existing entries). Group additions under a `# Reorg 2026-06-19` comment:

```gitignore
# Reorg 2026-06-19
__pycache__/
*.py[cod]
*.sw[op]
.DS_Store
*.egg-info/
.pytest_cache/
.venv/
outputs/
logs/
!logs/.gitkeep
.data/
.notes/
.plans/
.claude/worktrees/
```

- [ ] **Step 3: Verify nothing tracked is now ignored**

```bash
git status --ignored | head -40
git ls-files | xargs -I{} git check-ignore -v "{}" 2>/dev/null
```

Expected: the second command prints nothing (no tracked file is now ignored). If something is, either the file should be removed from tracking (`git rm --cached <file>`) or the pattern is too broad.

- [ ] **Step 4: Verify baseline**

```bash
pytest --collect-only -q 2>&1 | tail -5
```

Expected: same `N` as Task 1 Step 1.

- [ ] **Step 5: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore __pycache__, egg-info, outputs, logs, notes, plans"
```

---

## Task 3: `docs/` Diátaxis migration

**Files:**
- Create: `docs/README.md`, `docs/tutorials/`, `docs/how-to/`, `docs/reference/`, `docs/explanation/`
- Move: every existing doc into its quadrant (mapping below)
- Modify: `README.md` (repo root) and `CONTRIBUTING.md` to update any doc references

**Interfaces:**
- Consumes: nothing.
- Produces: stable doc paths consumed by Task 4+ (none directly), and by the spec in `docs/superpowers/specs/`.

- [ ] **Step 1: Create quadrant directories**

```bash
mkdir -p docs/tutorials docs/how-to docs/reference docs/explanation
```

- [ ] **Step 2: Move files into quadrants**

```bash
# Tutorials
git mv docs/tutorial-getting-started.md docs/tutorials/getting-started.md

# How-to
git mv docs/evaluating_models.md docs/how-to/evaluate-models.md
git mv docs/contribute_tasks.md docs/how-to/contribute-tasks.md
git mv docs/contributing_providers.md docs/how-to/contribute-providers.md
git mv docs/CLI_DATASET_USAGE.md docs/how-to/use-cli-datasets.md
git mv docs/testing.md docs/how-to/run-tests.md

# Reference
git mv docs/architecture.md docs/reference/architecture.md          # the "what" doc
git mv docs/reference-cli.md docs/reference/cli.md
git mv docs/reference-config.md docs/reference/config.md
git mv docs/reference-tasks.md docs/reference/tasks.md
git mv docs/reference-output-artifacts.md docs/reference/output-artifacts.md
git mv docs/DATASET_SPEC.md docs/reference/dataset-spec.md
git mv docs/benchy_probe_contract.md docs/reference/probe-contract.md
git mv docs/SCORING.md docs/reference/scoring.md
git mv docs/GENERATION_CONFIG.md docs/reference/generation-config.md
git mv docs/HANDLER_SYSTEM_GUIDE.md docs/reference/handler-system.md

# Explanation
git mv docs/explanation-architecture.md docs/explanation/architecture.md
git mv docs/VLLM_VERSION_MANAGEMENT.md docs/explanation/vllm-version-management.md

# Assets (benchy_2.png stays where the README points; the SVGs already moved in Task 1)
mkdir -p docs/assets
git mv docs/benchy_2.png docs/assets/benchy_2.png
```

- [ ] **Step 3: Create `docs/README.md` index**

Write to `docs/README.md`:

```markdown
# Benchy Documentation

This documentation follows the [Diátaxis](https://diataxis.fr/) framework.
Every doc lives in exactly one of four quadrants:

| Folder | When to read | When to write here |
|---|---|---|
| [`tutorials/`](./tutorials/) | New here — teach me by doing | You're explaining concepts to a beginner |
| [`how-to/`](./how-to/) | "How do I X?" — solve one task | You're documenting a recipe |
| [`reference/`](./reference/) | Look up exact behavior / schema | You're describing the system precisely |
| [`explanation/`](./explanation/) | Understand why it's designed this way | You're justifying a design decision |

## Conventions

- File names: lowercase, dash-separated, no prefix. The folder names the quadrant.
- If you can't decide which quadrant a doc fits, it probably doesn't belong in `docs/`.
- Design specs and implementation plans live in `superpowers/specs/` and `superpowers/plans/`.

## Index

### Tutorials
- [Getting started](./tutorials/getting-started.md)

### How-to
- [Evaluate models](./how-to/evaluate-models.md)
- [Contribute a task](./how-to/contribute-tasks.md)
- [Contribute a provider](./how-to/contribute-providers.md)
- [Use the CLI with datasets](./how-to/use-cli-datasets.md)
- [Run tests](./how-to/run-tests.md)

### Reference
- [Architecture (what)](./reference/architecture.md)
- [CLI](./reference/cli.md)
- [Config format](./reference/config.md)
- [Tasks](./reference/tasks.md)
- [Output artifacts](./reference/output-artifacts.md)
- [Dataset spec](./reference/dataset-spec.md)
- [Probe contract](./reference/probe-contract.md)
- [Scoring](./reference/scoring.md)
- [Generation config](./reference/generation-config.md)
- [Handler system](./reference/handler-system.md)

### Explanation
- [Architecture (why)](./explanation/architecture.md)
- [vLLM version management](./explanation/vllm-version-management.md)
```

- [ ] **Step 4: Rewrite doc-path references in README and CONTRIBUTING**

Generate the mapping then sed-sweep. For each `(old, new)` pair, run:

```bash
# Repeat per pair:
sed -i.bak 's|docs/tutorial-getting-started.md|docs/tutorials/getting-started.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/evaluating_models.md|docs/how-to/evaluate-models.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/contribute_tasks.md|docs/how-to/contribute-tasks.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/contributing_providers.md|docs/how-to/contribute-providers.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/CLI_DATASET_USAGE.md|docs/how-to/use-cli-datasets.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/testing.md|docs/how-to/run-tests.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/reference-cli.md|docs/reference/cli.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/reference-config.md|docs/reference/config.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/reference-tasks.md|docs/reference/tasks.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/reference-output-artifacts.md|docs/reference/output-artifacts.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/DATASET_SPEC.md|docs/reference/dataset-spec.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/benchy_probe_contract.md|docs/reference/probe-contract.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/SCORING.md|docs/reference/scoring.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/GENERATION_CONFIG.md|docs/reference/generation-config.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/HANDLER_SYSTEM_GUIDE.md|docs/reference/handler-system.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/architecture.md|docs/reference/architecture.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/explanation-architecture.md|docs/explanation/architecture.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/VLLM_VERSION_MANAGEMENT.md|docs/explanation/vllm-version-management.md|g' README.md CONTRIBUTING.md
sed -i.bak 's|docs/benchy_2.png|docs/assets/benchy_2.png|g' README.md CONTRIBUTING.md
rm README.md.bak CONTRIBUTING.md.bak
```

NOTE: `architecture.md` appears in both the `reference` move and the `explanation` source — the order above is correct (process `architecture.md` after `explanation-architecture.md`, since the explanation version's reference inside the text might still mention the old `architecture.md` path).

- [ ] **Step 5: Manually inspect README diff**

```bash
git diff README.md CONTRIBUTING.md | head -120
```

Expected: only path strings changed. If anything else was modified (e.g. a sed command over-matched), revert and rerun with more precise patterns.

- [ ] **Step 6: Sed-sweep doc-path references inside docs themselves**

```bash
# Repeat the same mapping over all .md files inside docs/
for f in docs/**/*.md; do
  sed -i.bak 's|docs/tutorial-getting-started.md|tutorials/getting-started.md|g' "$f"
  # ...repeat for every mapping, this time with the docs/-relative new path
done
find docs -name '*.bak' -delete
```

(Inside docs/, references are relative — drop the leading `docs/` when rewriting.)

- [ ] **Step 7: Verify baseline**

```bash
pytest --collect-only -q 2>&1 | tail -5
```

Expected: unchanged from Task 2.

- [ ] **Step 8: Commit**

```bash
git add -A docs/ README.md CONTRIBUTING.md
git commit -m "docs: complete Diátaxis migration

Sort every doc into tutorials/, how-to/, reference/, or explanation/.
Add docs/README.md as index. Rewrite all doc-path references in
README.md, CONTRIBUTING.md, and inside docs/ itself."
```

---

## Task 4: `tests/` mirror reshape

**Files:**
- Create: `tests/cli/`, `tests/core/`, `tests/engine/`, `tests/interfaces/`, `tests/tasks/handlers/`, `tests/tasks/common/`, `tests/tasks/transcription/`, `tests/tasks/translation/`, `tests/tasks/document_extraction/` (each with `__init__.py`)
- Move: 36 test files into new subdirs (mapping below)

**Interfaces:**
- Consumes: nothing.
- Produces: tests that still collect with the exact same count.

- [ ] **Step 1: Baseline test count**

```bash
pytest --collect-only -q 2>&1 | tail -3
```

Note the count `N`.

- [ ] **Step 2: Create subdirs with empty `__init__.py`**

```bash
mkdir -p tests/cli tests/core tests/engine tests/interfaces \
         tests/tasks/handlers tests/tasks/common \
         tests/tasks/transcription tests/tasks/translation \
         tests/tasks/document_extraction
for d in tests/cli tests/core tests/engine tests/interfaces \
         tests/tasks tests/tasks/handlers tests/tasks/common \
         tests/tasks/transcription tests/tasks/translation \
         tests/tasks/document_extraction; do
  touch "$d/__init__.py"
done
```

- [ ] **Step 3: Move test files (exact mapping)**

```bash
# cli/
git mv tests/test_benchy_cli_probe.py tests/cli/
git mv tests/test_cli_arguments.py tests/cli/

# core/
git mv tests/test_config_loader.py tests/core/
git mv tests/test_outcome.py tests/core/
git mv tests/test_retry.py tests/core/
git mv tests/test_protocols.py tests/core/

# engine/
git mv tests/test_benchmark_runner.py tests/engine/
git mv tests/test_task_status_checkpoint.py tests/engine/
git mv tests/test_connection.py tests/engine/
git mv tests/test_probe_runner.py tests/engine/
git mv tests/test_pipeline_summary.py tests/engine/

# interfaces/
git mv tests/test_openai_audio_interface.py tests/interfaces/
git mv tests/test_transformers_audio_interface.py tests/interfaces/
git mv tests/test_surus_factura_interface.py tests/interfaces/

# tasks/handlers/
git mv tests/test_handlers_base.py tests/tasks/handlers/
git mv tests/test_handlers_freeform.py tests/tasks/handlers/
git mv tests/test_handlers_structured.py tests/tasks/handlers/
git mv tests/test_handlers_multiple_choice.py tests/tasks/handlers/
git mv tests/test_multimodal_structured.py tests/tasks/handlers/

# tasks/common/
git mv tests/test_dataset_loaders.py tests/tasks/common/
git mv tests/test_dataset_adapters.py tests/tasks/common/
git mv tests/test_dataset_utils.py tests/tasks/common/
git mv tests/test_audio_preprocessing.py tests/tasks/common/
git mv tests/test_image_preprocessing.py tests/tasks/common/
git mv tests/test_config_generator.py tests/tasks/common/
git mv tests/test_metrics_choice.py tests/tasks/common/
git mv tests/test_metrics_extended.py tests/tasks/common/
git mv tests/test_translation_metrics.py tests/tasks/common/
git mv tests/test_wer_cer_metrics.py tests/tasks/common/
git mv tests/test_field_diagnostics_report.py tests/tasks/common/
git mv tests/test_structured_metrics_calculator.py tests/tasks/common/
git mv tests/test_task_config_schema.py tests/tasks/common/
git mv tests/test_registry_adhoc.py tests/tasks/common/
git mv tests/test_group_runner.py tests/tasks/common/

# tasks/transcription/
git mv tests/test_transcription_handler.py tests/tasks/transcription/
git mv tests/test_transcription_capabilities.py tests/tasks/transcription/
git mv tests/test_fleurs_subtasks.py tests/tasks/transcription/

# tasks/translation/
git mv tests/test_translation_handler_metrics.py tests/tasks/translation/

# tasks/document_extraction/
git mv tests/test_facturas_argentinas_dataset_id.py tests/tasks/document_extraction/

# leaderboard/ already exists, files stay
```

`tests/conftest.py` and `tests/fixtures.py` stay at the top of `tests/` so they apply to all subdirs.

- [ ] **Step 4: Re-collect and compare**

```bash
pytest --collect-only -q 2>&1 | tail -3
```

Expected: same `N` as Step 1. If different, a test failed to be moved or `__init__.py` is missing somewhere — fix before continuing.

- [ ] **Step 5: Run the full suite**

```bash
pytest -q 2>&1 | tail -10
```

Expected: same pass/fail tallies as before the reshape.

- [ ] **Step 6: Commit**

```bash
git add -A tests/
git commit -m "chore: mirror tests/ structure against src/ subpackages

36 test files moved into tests/{cli,core,engine,interfaces,tasks/...}.
Same collection count; same pass/fail outcomes. conftest.py and
fixtures.py stay at tests/ root."
```

---

## Task 5: Extract `src/cli/`

**Files:**
- Create: `src/cli/__init__.py`, `src/cli/main.py`, `src/cli/eval.py`, `src/cli/probe.py`
- Move: `src/benchy_cli.py` → `src/cli/main.py`, `src/benchy_cli_eval.py` → `src/cli/eval.py`, `src/benchy_cli_probe.py` → `src/cli/probe.py`
- Shim: `src/benchy_cli.py`, `src/benchy_cli_eval.py`, `src/benchy_cli_probe.py` (3 shims)
- Modify: `pyproject.toml` (`[project.scripts]` entry)
- Modify: `tests/cli/test_benchy_cli_probe.py`, `tests/cli/test_cli_arguments.py` (update import paths to the new canonical location, not the shim — exercises the real module)

**Interfaces:**
- Consumes: shim policy.
- Produces: `src.cli.main:main`, `src.cli.eval:*`, `src.cli.probe:*`. Old `src.benchy_cli*` paths continue to import-work via shims.

- [ ] **Step 1: Create `src/cli/` package**

```bash
mkdir -p src/cli
touch src/cli/__init__.py
```

- [ ] **Step 2: Move the three CLI modules**

```bash
git mv src/benchy_cli.py src/cli/main.py
git mv src/benchy_cli_eval.py src/cli/eval.py
git mv src/benchy_cli_probe.py src/cli/probe.py
```

- [ ] **Step 3: Update internal imports inside the moved files**

The moved files may reference each other (e.g. `main.py` likely imports from `eval.py` and `probe.py`). Grep:

```bash
grep -n "from src\.benchy_cli\|from src import benchy_cli\|from \.benchy_cli" src/cli/*.py
```

For each match, rewrite:
- `from src.benchy_cli_eval import X` → `from src.cli.eval import X`
- `from src.benchy_cli_probe import X` → `from src.cli.probe import X`
- `from src.benchy_cli import X` → `from src.cli.main import X`

Verify with a follow-up `grep` — no `src.benchy_cli*` references inside `src/cli/`.

- [ ] **Step 4: Add shims at old paths**

Create `src/benchy_cli.py`:

```python
# src/benchy_cli.py — DEPRECATED: moved to src.cli.main.
# Remove this shim after one release cycle.
from src.cli.main import *  # noqa: F401, F403
from src.cli.main import main  # explicit re-export for entry_point compatibility  # noqa: F401
```

Create `src/benchy_cli_eval.py`:

```python
# src/benchy_cli_eval.py — DEPRECATED: moved to src.cli.eval.
# Remove this shim after one release cycle.
from src.cli.eval import *  # noqa: F401, F403
```

Create `src/benchy_cli_probe.py`:

```python
# src/benchy_cli_probe.py — DEPRECATED: moved to src.cli.probe.
# Remove this shim after one release cycle.
from src.cli.probe import *  # noqa: F401, F403
from src.cli.probe import _build_connection_info  # private name still imported by tests  # noqa: F401
```

(The `_build_connection_info` line is included because `tests/cli/test_benchy_cli_probe.py` imports it explicitly; star-import does not re-export underscore-prefixed names.)

- [ ] **Step 5: Update `pyproject.toml`**

Find:

```toml
[project.scripts]
benchy = "src.benchy_cli:main"
```

Replace with:

```toml
[project.scripts]
benchy = "src.cli.main:main"
```

- [ ] **Step 6: Update test imports to canonical paths**

In `tests/cli/test_benchy_cli_probe.py`, change:

```python
from src.benchy_cli_probe import _build_connection_info
```

to:

```python
from src.cli.probe import _build_connection_info
```

In `tests/cli/test_cli_arguments.py`, change:

```python
from src.benchy_cli_eval import (
```

to:

```python
from src.cli.eval import (
```

(Tests should exercise the canonical path, not the shim. The shim's behavior is verified separately in Step 7.)

- [ ] **Step 7: Verify shims and CLI entry point**

```bash
# Reinstall to pick up pyproject.toml change
pip install -e . --quiet
benchy --help | head -5
```

Expected: prints the benchy CLI help. If it fails with `ModuleNotFoundError`, the entry point change didn't propagate — re-run `pip install -e .`.

```bash
# Verify shims still work
python -c "from src.benchy_cli import main; print(main)"
python -c "from src.benchy_cli_eval import *; print('eval shim OK')"
python -c "from src.benchy_cli_probe import _build_connection_info; print(_build_connection_info)"
```

Expected: each prints without error.

- [ ] **Step 8: Run tests**

```bash
pytest tests/cli -q
pytest -q 2>&1 | tail -5
```

Expected: tests/cli passes; full suite same as Task 4 Step 5.

- [ ] **Step 9: Smoke run**

```bash
benchy eval --tasks spanish.copa_es --provider openai --model-name gpt-4o-mini \
  --limit 2 --run-id reorg_smoke_t5 --exit-policy smoke
echo "exit=$?"
```

Expected: `exit=0`. (Substitute config if no OpenAI key; record in commit message.)

- [ ] **Step 10: Commit**

```bash
git add -A src/ pyproject.toml tests/cli/
git commit -m "refactor: extract src/cli/ subpackage from loose src root modules

Move benchy_cli{,_eval,_probe}.py into src/cli/{main,eval,probe}.py.
Update pyproject.toml entry point to src.cli.main:main. Leave shims
at old paths for one release cycle. Tests import canonical paths."
```

---

## Task 6: Extract `src/core/`

**Files:**
- Create: `src/core/__init__.py` and 10 new module files inside `src/core/`
- Move: 10 modules from `src/` root into `src/core/`
- Shim: 10 shims at old paths
- Modify: every importer in `src/`, `tests/`, `scripts/` of any moved module

**Interfaces:**
- Consumes: shim policy (Task 5).
- Produces: canonical paths `src.core.<name>` for all 10 modules. Old `src.<name>` paths continue to work via shims.

Modules moving (10): `config_loader.py`, `config_manager.py`, `generation_config.py`, `gpu_config.py`, `logging_utils.py`, `outcome.py`, `run_id_manager.py`, `signal_utils.py`, `task_completion_checker.py`, `prefect_compat.py`.

- [ ] **Step 1: Create `src/core/` package**

```bash
mkdir -p src/core
touch src/core/__init__.py
```

- [ ] **Step 2: Move modules**

```bash
for m in config_loader config_manager generation_config gpu_config \
         logging_utils outcome run_id_manager signal_utils \
         task_completion_checker prefect_compat; do
  git mv "src/$m.py" "src/core/$m.py"
done
```

- [ ] **Step 3: Find every importer of these modules**

```bash
grep -rn "from src\.\(config_loader\|config_manager\|generation_config\|gpu_config\|logging_utils\|outcome\|run_id_manager\|signal_utils\|task_completion_checker\|prefect_compat\)" \
  --include='*.py' src tests scripts
```

Save the list. For each match, rewrite the import:

- `from src.config_loader import X` → `from src.core.config_loader import X`
- `from src.config_manager import X` → `from src.core.config_manager import X`
- (… same pattern for each of the 10 modules)

Do this with `sed`:

```bash
for m in config_loader config_manager generation_config gpu_config \
         logging_utils outcome run_id_manager signal_utils \
         task_completion_checker prefect_compat; do
  grep -rl "from src\.$m" --include='*.py' src tests scripts | \
    xargs -I{} sed -i.bak "s|from src\.$m|from src.core.$m|g" {}
done
find . -name '*.py.bak' -delete
```

- [ ] **Step 4: Handle internal imports inside the moved modules**

After moving, some modules might import each other. For example `config_manager.py` might `from src.config_loader import …`. The sed sweep in Step 3 also rewrote those, but they should reference `src.core.<name>` now. Verify:

```bash
grep -n "from src\." src/core/*.py
```

Expected: any cross-references between core modules say `from src.core.<name>`, not `from src.<name>`. Fix any that slipped.

Cross-package imports (e.g. `from src.engine.protocols import ...`) are fine as-is.

- [ ] **Step 5: Create shims at old paths**

For each of the 10 modules, create a shim. Example for `outcome.py`:

```python
# src/outcome.py — DEPRECATED: moved to src.core.outcome.
# Remove this shim after one release cycle.
from src.core.outcome import *  # noqa: F401, F403
```

Some modules have names not re-exported by `*` (anything `_underscore_prefixed`). Check each module's `__all__` or grep for underscore-prefixed public-ish names imported elsewhere:

```bash
grep -rn "from src\.outcome import _" --include='*.py' .
```

For each underscore name imported externally, add an explicit re-export line in the shim. Repeat for the other 9 modules. Likely only `outcome` and `pipeline` (Task 7) have any.

- [ ] **Step 6: Verify imports resolve at both paths**

```bash
python -c "import src.core.outcome; import src.outcome; print('OK')"
python -c "from src.core.config_loader import load_config; from src.config_loader import load_config as legacy; assert load_config is legacy; print('OK')"
```

Expected: prints `OK` for each. If `is legacy` assertion fails, the shim is re-exporting a different object — fix the shim.

- [ ] **Step 7: Run tests**

```bash
pytest -q 2>&1 | tail -10
```

Expected: same pass/fail tallies as Task 5.

- [ ] **Step 8: Smoke run**

```bash
benchy eval --tasks spanish.copa_es --provider openai --model-name gpt-4o-mini \
  --limit 2 --run-id reorg_smoke_t6 --exit-policy smoke
echo "exit=$?"
```

Expected: `exit=0`.

- [ ] **Step 9: Commit**

```bash
git add -A src/
git commit -m "refactor: extract src/core/ subpackage from loose src root modules

Move 10 cross-cutting modules (config_loader, config_manager,
generation_config, gpu_config, logging_utils, outcome, run_id_manager,
signal_utils, task_completion_checker, prefect_compat) into src/core/.
Update all importers. Shims at old paths for one release cycle."
```

---

## Task 7: Move `src/pipeline.py` into `src/engine/`

**Files:**
- Move: `src/pipeline.py` → `src/engine/pipeline.py`
- Shim: `src/pipeline.py`
- Modify: every importer

**Interfaces:**
- Consumes: shim policy.
- Produces: canonical path `src.engine.pipeline`. Old `src.pipeline` continues to work via shim.

- [ ] **Step 1: Move pipeline**

```bash
git mv src/pipeline.py src/engine/pipeline.py
```

- [ ] **Step 2: Find and rewrite importers**

```bash
grep -rn "from src\.pipeline\|import src\.pipeline" --include='*.py' src tests scripts
grep -rl "from src\.pipeline" --include='*.py' src tests scripts | \
  xargs -I{} sed -i.bak 's|from src\.pipeline|from src.engine.pipeline|g' {}
find . -name '*.py.bak' -delete
```

- [ ] **Step 3: Create shim at `src/pipeline.py`**

```python
# src/pipeline.py — DEPRECATED: moved to src.engine.pipeline.
# Remove this shim after one release cycle.
from src.engine.pipeline import *  # noqa: F401, F403
# Re-export private name imported by tests:
from src.engine.pipeline import _summarize_single_task_metrics  # noqa: F401
```

(The `_summarize_single_task_metrics` line is needed because `tests/engine/test_pipeline_summary.py` imports it explicitly.)

- [ ] **Step 4: Verify**

```bash
python -c "from src.pipeline import _summarize_single_task_metrics; print(_summarize_single_task_metrics)"
python -c "from src.engine.pipeline import _summarize_single_task_metrics; print('canonical OK')"
pytest tests/engine/test_pipeline_summary.py -q
```

Expected: prints function, then `canonical OK`, then test passes.

- [ ] **Step 5: Full test + smoke**

```bash
pytest -q 2>&1 | tail -5
benchy eval --tasks spanish.copa_es --provider openai --model-name gpt-4o-mini \
  --limit 2 --run-id reorg_smoke_t7 --exit-policy smoke
echo "exit=$?"
```

Expected: tests same as Task 6; `exit=0`.

- [ ] **Step 6: Commit**

```bash
git add -A src/
git commit -m "refactor: move src/pipeline.py into src/engine/

pipeline is part of the engine subsystem (Prefect orchestration,
vLLM lifecycle). Update importers. Shim at src/pipeline.py for one
release cycle."
```

---

## Task 8: Dedup `src/common/` into `src/tasks/common/utils/`

**Files:**
- Diff: 2 duplicate pairs (`choice_utils.py`, `dataset_utils.py`) — `src/common/` vs `src/tasks/common/utils/`
- Move: `src/common/schema_sanitizer.py` and `src/common/summary_reporter.py` (no duplicates) → `src/tasks/common/utils/`
- Resolve: duplicates — keep `src/tasks/common/utils/` copies after confirming or merging
- Shim: `src/common/{choice_utils,dataset_utils,schema_sanitizer,summary_reporter}.py`
- Delete: `src/common/` directory (after all moves)
- Modify: every importer

**Interfaces:**
- Consumes: nothing.
- Produces: `src.tasks.common.utils.<name>` as the canonical path for all four modules. `src.common.<name>` shims for back-compat.

- [ ] **Step 1: Diff the duplicates**

```bash
diff -u src/common/choice_utils.py src/tasks/common/utils/choice_utils.py
diff -u src/common/dataset_utils.py src/tasks/common/utils/dataset_utils.py
```

- If a diff is empty: the `src/tasks/common/utils/` copy is canonical. Skip to Step 2 for that file.
- If a diff is non-empty: read both versions carefully. Pick the newer / more correct one as canonical (check `git log src/common/<file>` vs `git log src/tasks/common/utils/<file>` for last-modified timestamps). If unsure, copy the loser into `misc/<filename>.alt` and surface to the user before continuing.

- [ ] **Step 2: Move the non-duplicates**

```bash
git mv src/common/schema_sanitizer.py src/tasks/common/utils/schema_sanitizer.py
git mv src/common/summary_reporter.py src/tasks/common/utils/summary_reporter.py
```

- [ ] **Step 3: Resolve duplicates**

For each of `choice_utils.py` and `dataset_utils.py`:
- If diff was empty in Step 1: `rm src/common/<file>` (`git rm` since it was tracked).
- If diff was non-empty: write the chosen canonical content into `src/tasks/common/utils/<file>` first, then `git rm src/common/<file>`.

```bash
git rm src/common/choice_utils.py
git rm src/common/dataset_utils.py
```

- [ ] **Step 4: Find and rewrite importers**

```bash
grep -rn "from src\.common\." --include='*.py' src tests scripts
grep -rl "from src\.common\." --include='*.py' src tests scripts | \
  xargs -I{} sed -i.bak 's|from src\.common\.|from src.tasks.common.utils.|g' {}
find . -name '*.py.bak' -delete
```

- [ ] **Step 5: Replace `src/common/__init__.py` with shims**

Reduce `src/common/` to a shim-only package. Write `src/common/__init__.py`:

```python
# src/common/ — DEPRECATED: contents moved to src.tasks.common.utils.
# Remove this package after one release cycle.
from src.tasks.common.utils.choice_utils import *  # noqa: F401, F403
from src.tasks.common.utils.dataset_utils import *  # noqa: F401, F403
from src.tasks.common.utils.schema_sanitizer import *  # noqa: F401, F403
from src.tasks.common.utils.summary_reporter import *  # noqa: F401, F403
```

Additionally create per-module shims so `from src.common.<module> import X` still works:

```bash
for m in choice_utils dataset_utils schema_sanitizer summary_reporter; do
  cat > "src/common/$m.py" <<EOF
# src/common/$m.py — DEPRECATED: moved to src.tasks.common.utils.$m.
# Remove this shim after one release cycle.
from src.tasks.common.utils.$m import *  # noqa: F401, F403
EOF
done
```

- [ ] **Step 6: Verify**

```bash
python -c "from src.common.choice_utils import *; print('shim OK')"
python -c "from src.tasks.common.utils.choice_utils import *; print('canonical OK')"
pytest -q 2>&1 | tail -5
```

Expected: both `OK`s; tests same as Task 7.

- [ ] **Step 7: Smoke run**

```bash
benchy eval --tasks spanish.copa_es --provider openai --model-name gpt-4o-mini \
  --limit 2 --run-id reorg_smoke_t8 --exit-policy smoke
echo "exit=$?"
```

Expected: `exit=0`.

- [ ] **Step 8: Commit**

```bash
git add -A src/
git commit -m "refactor: dedupe src/common/ into src/tasks/common/utils/

src/common/{choice_utils,dataset_utils} were duplicated in
tasks/common/utils/. Consolidate into one canonical home under
tasks/common/utils/. schema_sanitizer.py and summary_reporter.py
move alongside. Shims at src/common/ for one release cycle."
```

---

## Task 9: Finish per-dataset task-folder migration (portuguese, translation)

**Files:**
- Diff: each flat task file against the corresponding `datasets/<name>/task.py`
- Delete: flat files where the folder version is canonical
- Promote: folder versions where the flat file is newer (copy newer content into the folder version first)
- Modify: any importers (registry, internal imports, tests)

**Interfaces:**
- Consumes: nothing.
- Produces: tasks discoverable only via `src.tasks.portuguese.datasets.<name>` and `src.tasks.translation.datasets.<name>`. Old flat paths gone.

Files to resolve (7 total):
- `src/tasks/portuguese/bluex.py` vs `src/tasks/portuguese/datasets/bluex/task.py`
- `src/tasks/portuguese/assin2_rte.py` vs `src/tasks/portuguese/datasets/assin2_rte/task.py`
- `src/tasks/portuguese/assin2_sts.py` vs `src/tasks/portuguese/datasets/assin2_sts/task.py`
- `src/tasks/portuguese/enem_challenge.py` vs `src/tasks/portuguese/datasets/enem_challenge/task.py`
- `src/tasks/portuguese/oab_exams.py` vs `src/tasks/portuguese/datasets/oab_exams/task.py`
- `src/tasks/translation/flores.py` vs `src/tasks/translation/datasets/flores/`
- `src/tasks/translation/opus.py` vs `src/tasks/translation/datasets/opus/`

- [ ] **Step 1: Inspect translation/datasets layout**

```bash
ls src/tasks/translation/datasets/flores src/tasks/translation/datasets/opus
```

Confirm a `task.py` (or equivalent main entry) exists in each. If not, the migration is incomplete on that side — instead of deleting the flat file, promote its content into `datasets/<name>/task.py` first.

- [ ] **Step 2: Diff each pair and decide direction**

For each of the 7 pairs:

```bash
# Example
diff -u src/tasks/portuguese/bluex.py src/tasks/portuguese/datasets/bluex/task.py | head -80
git log -1 --format='%cI %h %s' src/tasks/portuguese/bluex.py
git log -1 --format='%cI %h %s' src/tasks/portuguese/datasets/bluex/task.py
```

Decision rule:
- If folder version is newer or has no missing logic → folder is canonical; delete flat.
- If flat is newer → copy flat content into folder version, then delete flat.
- If neither is clearly canonical and the diff is non-trivial → copy the flat version into `misc/<task>.flat.py` for review and proceed with the folder version. Note the demotion in the commit message.

Record decisions in a temporary scratch file (e.g. `misc/task9-decisions.md`) before deleting anything.

- [ ] **Step 3: Apply decisions — promote where needed**

For pairs where flat is canonical, before deleting:

```bash
cp src/tasks/portuguese/bluex.py src/tasks/portuguese/datasets/bluex/task.py
```

For pairs where folder is canonical, no copy needed.

- [ ] **Step 4: Delete flat files**

```bash
git rm src/tasks/portuguese/bluex.py
git rm src/tasks/portuguese/assin2_rte.py
git rm src/tasks/portuguese/assin2_sts.py
git rm src/tasks/portuguese/enem_challenge.py
git rm src/tasks/portuguese/oab_exams.py
git rm src/tasks/translation/flores.py
git rm src/tasks/translation/opus.py
```

- [ ] **Step 5: Check task registry still discovers them**

```bash
benchy tasks --json | python -c "import json, sys; data = json.load(sys.stdin); tasks = list(data.get('tasks', data)) if isinstance(data, dict) else data; print([t for t in tasks if any(name in str(t) for name in ['bluex', 'assin2', 'enem', 'oab', 'flores', 'opus'])])"
```

Expected: every migrated task name appears. If `benchy tasks` lacks a `--json` flag, fall back to:

```bash
benchy tasks 2>&1 | grep -E 'bluex|assin2|enem|oab|flores|opus'
```

Expected: every migrated task name appears.

If a task is missing, the registry expected the flat file. Open `src/tasks/registry.py` and inspect the discovery logic — likely a glob that needs to also match `datasets/*/task.py`. Patch if so; otherwise restore the flat file and surface to the user.

- [ ] **Step 6: Find and rewrite importers**

```bash
grep -rn "from src\.tasks\.portuguese\.\(bluex\|assin2_rte\|assin2_sts\|enem_challenge\|oab_exams\)\|from src\.tasks\.translation\.\(flores\|opus\)" \
  --include='*.py' src tests scripts
```

For each match, rewrite. Example:
- `from src.tasks.portuguese.bluex import X` → `from src.tasks.portuguese.datasets.bluex.task import X`
- `from src.tasks.translation.flores import X` → `from src.tasks.translation.datasets.flores.task import X` (or whatever the actual entry is)

```bash
for old in bluex assin2_rte assin2_sts enem_challenge oab_exams; do
  grep -rl "from src\.tasks\.portuguese\.$old" --include='*.py' src tests scripts | \
    xargs -I{} sed -i.bak "s|from src\.tasks\.portuguese\.$old|from src.tasks.portuguese.datasets.$old.task|g" {}
done
for old in flores opus; do
  grep -rl "from src\.tasks\.translation\.$old" --include='*.py' src tests scripts | \
    xargs -I{} sed -i.bak "s|from src\.tasks\.translation\.$old|from src.tasks.translation.datasets.$old.task|g" {}
done
find . -name '*.py.bak' -delete
```

- [ ] **Step 7: Run tests + smoke**

```bash
pytest -q 2>&1 | tail -10
```

Expected: same pass/fail tallies as Task 8.

If a Portuguese or translation task is part of the smoke suite, run a portuguese-specific smoke:

```bash
benchy eval --tasks portuguese.bluex --provider openai --model-name gpt-4o-mini \
  --limit 2 --run-id reorg_smoke_t9 --exit-policy smoke
echo "exit=$?"
```

Expected: `exit=0`. (Substitute task if `bluex` requires multimodal capabilities the smoke config doesn't have.)

- [ ] **Step 8: Commit**

```bash
git add -A src/ misc/
git commit -m "refactor: finish per-dataset folder layout for portuguese + translation

Delete duplicate flat tasks/<group>/<name>.py files now that
tasks/<group>/datasets/<name>/task.py is canonical. Promote newer
flat content where applicable. Update importers."
```

---

## Task 10: Rename `configs/tests/` → `configs/examples/`

**Files:**
- Move: `configs/tests/` → `configs/examples/`
- Modify: any reference to `configs/tests/` (docs, scripts, agent skills)

**Interfaces:**
- Consumes: nothing.
- Produces: clearer config taxonomy. No behavior change.

- [ ] **Step 1: Move the directory**

```bash
git mv configs/tests configs/examples
```

- [ ] **Step 2: Find and rewrite references**

```bash
grep -rn "configs/tests" --include='*.py' --include='*.yaml' --include='*.yml' \
  --include='*.md' --include='*.sh' . 2>/dev/null | grep -v '\.git/\|\.venv/'
```

For each match, replace `configs/tests` with `configs/examples`:

```bash
grep -rl "configs/tests" --include='*.py' --include='*.yaml' --include='*.yml' \
  --include='*.md' --include='*.sh' . 2>/dev/null | grep -v '\.git/\|\.venv/' | \
  xargs -I{} sed -i.bak 's|configs/tests|configs/examples|g' {}
find . -name '*.bak' -not -path './.git/*' -delete
```

- [ ] **Step 3: Verify with a smoke run**

```bash
# If a configs/examples/ file is the smoke config, exercise it.
ls configs/examples/
benchy eval --config configs/examples/spanish-gptoss.yaml \
  --tasks spanish.copa_es --limit 2 --run-id reorg_smoke_t10 --exit-policy smoke 2>&1 | tail -5
echo "exit=$?"
```

Expected: `exit=0` or whatever the config naturally produces. If the smoke config requires capabilities you can't provide, substitute with a config you can — record substitution in commit message.

- [ ] **Step 4: Full test pass**

```bash
pytest -q 2>&1 | tail -5
```

Expected: same as Task 9.

- [ ] **Step 5: Commit**

```bash
git add -A configs/ docs/ scripts/ .agent/
git commit -m "chore: rename configs/tests/ to configs/examples/

The directory holds example configs, not test fixtures. New name
removes ambiguity."
```

---

## Post-implementation

After Task 10 lands:

1. **Sweep open shims** — create a `misc/MOVES.md` cataloguing every shim added (Tasks 5–8). One line per shim: `<old path> -> <new path> (added 2026-06-19, remove after vX.Y.Z)`. Commit as `docs: track shim cleanup queue`.

2. **PR description** — when opening the PR (if the user opts in), summarize each task as one bullet pointing to its commit hash. Note the single externally visible change: `pyproject.toml` entry point.

3. **Soak period** — leave `misc/` populated for at least 30 days before any follow-up deletion PR. Use the time to confirm no downstream consumer (shell scripts, other agents, dashboards) was relying on the moved paths.

4. **Open implementation-time questions surfaced during execution** (track answers in commit messages or `misc/`):
   - Should `src/__init__.py` add `from . import cli, core, engine, ...`? Default: no (keep empty) unless tests require it.
   - For Task 9 promotions, also propagate the metadata.yaml / `__init__.py` shape, or just the Python content? Default: Python only; folder structure already exists.
   - `misc/MOVES.md` location — `misc/` (visible triage) vs `docs/superpowers/specs/` (alongside this design)? Default: `misc/MOVES.md`.
