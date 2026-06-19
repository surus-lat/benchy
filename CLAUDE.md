# Benchy — Agent Guide

## Skills

- **`oracle-plan`** — Bidirectional algorithm for converting between a working
  implementation and a battle-tested design document. Use when extracting a plan
  from a branch (`git diff` → design doc → GitHub issue) or executing a plan to
  a working implementation (design doc → TDD → verified feature). The full
  reference instance is `feat/transcription-support` / `surus-lat/benchy#31`.

Skills live in `.agent/skills/<name>/SKILL.md`. Invoke via the `Skill` tool.

## Git commits

Never add `Co-Authored-By` trailers to commit messages.
