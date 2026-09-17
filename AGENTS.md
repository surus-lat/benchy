# Benchy — Agent Guide

See [`CLAUDE.md`](./CLAUDE.md). It is the single agent-facing guide for this repo and
is kept current; this file exists only so tools that look for `AGENTS.md` find it.

Quick orientation:

- The engine is `benchy/` — ten files, ~780 lines, PyYAML the only dependency.
- `README.md` is short and accurate. Read it first.
- Gate: `python -m pytest tests -q` and `python -m ruff check benchy tests`.
- Normative documents live in `paper/`; design decisions in `docs/engine-v1/PLAN.md`.
