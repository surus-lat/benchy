# Benchy — Agent Guide

## Skills

Skills live in `.agent/skills/<name>/SKILL.md`. Invoke via the `Skill` tool.

See [`.agent/skills/README.md`](./.agent/skills/README.md) for the full
index, grouped by purpose: running benchmarks, authoring a benchmark spec,
extending benchy, and specialized benchmarks (Whisper-family ASR).

Highlights:

- **`run-benchmark`** — End-to-end wrapper: validate → smoke → full → results.
- **`evaluate`** — Run `benchy eval` directly with the canonical smoke→full pattern.
- **`whisper-benchmark`** — Local FLEURS ASR panel via `transformers_audio` on Mac.
- **`oracle-plan`** — Bidirectional plan ↔ implementation algorithm. Reference
  instance: `feat/transcription-support` / `surus-lat/benchy#31`.

## Git commits

Never add `Co-Authored-By` trailers to commit messages.
