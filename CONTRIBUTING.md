# Contributing to Benchy

Thanks for contributing to Benchy, the LATAM Leaderboard benchmarking suite. We welcome
new tasks, provider integrations, and documentation improvements.

## Project Goals

- Evaluate AI systems, not just models (LLMs and task-specific endpoints).
- Keep tasks and interfaces decoupled so each can evolve independently.
- Make it easy to add LATAM-focused evaluations.

## Where to Start

- Tasks: `docs/contribute_tasks.md`
- Providers: `docs/contributing_providers.md`
- Model/system evaluation: `docs/evaluating_models.md`

## Development Setup

```bash
uv sync
source .venv/bin/activate
```

Optional: `bash setup.sh` to prefetch structured extraction data.

## Contribution Workflow

1. Open an issue or start a discussion for larger changes.
2. Create a feature branch.
3. Keep PRs focused and include the tests you ran.

## What to Update

- New tasks should include a task config (`task.json`), task code, and a pipeline registration.
- New providers should include a provider config, an interface, and documentation updates.
- Docs should stay in sync with behavior in `src/` and `configs/`.

## Quick Validation

```bash
python eval.py --config configs/tests/spanish-gptoss.yaml --limit 2
```
