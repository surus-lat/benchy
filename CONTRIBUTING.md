# Contributing to Benchy

Thanks for contributing. Benchy is a semantic language and execution engine for
benchmarking AI programs.

## Project Goals

- Evaluate AI-systems, not just models: a model, a model with a prompt, a composed
  workflow and an agent are the same kind of thing from outside.
- Make it easy to *create* the benchmark that represents a problem, not just to run
  benchmarks that already exist.
- Keep the engine small enough to hold in your head.

## Where to Start

- [`README.md`](./README.md) — what benchy is and how to run it
- [`paper/technical-paper-v10.3.md`](./paper/technical-paper-v10.3.md) — the semantics
- [`paper/benchy-engine-spec-v1.2.md`](./paper/benchy-engine-spec-v1.2.md) — normative rules
- [`docs/engine-v1/PLAN.md`](./docs/engine-v1/PLAN.md) — design decisions, and the parts
  deliberately *not* built. Read this before adding a module, class or abstraction.

## Development Setup

```bash
uv venv --python 3.12
uv sync
source .venv/bin/activate
```

Optional: `bash setup.sh` to prefetch structured extraction data (or `BENCHY_SKIP_DATASET=1 bash setup.sh` to skip).

## Contribution Workflow

1. Open an issue or start a discussion for larger changes.
2. Create a feature branch.
3. Keep PRs focused and include the tests you ran.

## What to Update

- New tasks should include `metadata.yaml` plus task code under `src/tasks/<group>/`.
- New providers should include a provider config, an interface, and documentation updates.
- Docs should stay in sync with behavior in `src/` and `configs/`.

## Quick Validation

```bash
benchy eval --config configs/tests/spanish-gptoss.yaml --limit 2
```
