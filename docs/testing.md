# Testing Guide

This file is the project-wide testing contract for Benchy contributors and agents.
Use it when adding modules, changing behavior, or expanding coverage.

## Goals

- Keep tests **small, deterministic, and cheap**.
- Prefer **module-level confidence** before end-to-end runs.
- Protect machine-facing contracts (`run_outcome.json`, `probe_report.json`, `task_status.json`).
- Avoid large, brittle mocking setups.

## Test Layers

### 1) Unit tests (required first)

Scope: pure functions and local logic.

Examples:
- Status/exit policy logic (`src/outcome.py`)
- Capability merging (`src/engine/connection.py`)
- Scoring/parsing helpers (`src/tasks/common/*`)
- Image preprocessing (`src/interfaces/common/image_preprocessing.py`)

Rules:
- No network calls.
- No external services.
- Use tiny in-memory inputs and `tmp_path` when file IO is needed.

### 2) Seam tests (required for orchestration modules)

Scope: module boundaries where multiple components interact.

Examples:
- `BenchmarkRunner` with fake task + fake interface
- `group_runner` skip/error/aggregate paths with monkeypatched dependencies

Rules:
- Replace external dependencies with simple fakes.
- Validate behavior and contracts, not implementation details.

### 3) Contract tests (required for artifacts)

Scope: machine-readable outputs and schema stability.

Examples:
- `run_outcome.json` required keys + status semantics
- `probe_report.json` required keys + mode/transport state semantics

Rules:
- Parse JSON artifacts, never assert on human logs.
- Keep contract checks strict on required fields and status vocabulary.

### 4) End-to-end smoke (minimal)

Scope: one tiny eval path to catch integration regressions.

Rules:
- Keep it short (`--limit 1` or similarly small).
- Use it as a sanity check, not as primary correctness proof.

## Directory and Naming

- Put tests under `tests/`.
- File naming: `test_<module_or_flow>.py`.
- Test naming: `test_<behavior>_<expected_result>`.

Examples:
- `tests/test_outcome.py`
- `tests/test_benchmark_runner.py`
- `tests/test_group_runner.py`

## Required Schema for New Module Work

When adding or changing module behavior, include:

1. **At least 1 unit test** for the new logic.
2. **At least 1 edge-case test** (invalid input, empty input, fallback, or error path).
3. **If orchestration changed:** at least 1 seam test for the integration boundary.
4. **If artifact JSON changed:** update/add contract tests for required keys and status logic.

## Practical Patterns

- Prefer **fakes** over deep mocks.
- Patch only boundary functions/classes (provider clients, file writers, subprocess/network calls).
- Assert on stable outputs (returned dicts, artifact files, status values).
- Avoid asserting on internal log strings.

## What Not to Do

- Don’t add broad snapshot tests for large JSON payloads.
- Don’t couple tests to exact timestamps or ordering unless contract requires it.
- Don’t depend on real provider endpoints for unit/seam tests.

## Running Tests

```bash
pytest -q
```

Optional with coverage:

```bash
coverage run -m pytest -q
coverage report -m
```

## Lint Workflow (Ruff)

Linting is part of CI and should pass before merging.

Run locally:

```bash
ruff check src tests
```

Safe autofix pass (no unsafe fixes):

```bash
ruff check --fix src tests
```

Recommended workflow for larger cleanup:

1. Fix in small batches (about 10 findings at a time).
2. Re-run lint after each batch.
3. Re-run tests after each batch:

```bash
pytest -q
```

Rules:
- Do not use `--unsafe-fixes` unless explicitly approved.
- Prefer manual review for any fix beyond obvious unused imports/variables or no-op f-strings.
- If a lint finding reveals a real bug (for example undefined names), fix it with targeted code changes and tests.

## Iterative Expansion Plan

1. Keep adding unit tests around changed logic first.
2. Add seam tests for high-churn orchestrators.
3. Add/extend contract tests when artifacts evolve.
4. Keep E2E smoke small and stable.

This incremental approach keeps maintenance low while increasing confidence over time.
