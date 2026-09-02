# SUMMARY — s07 datacentric (search/s07, final)

## Final golem report (verbatim)
```json
{"cycles": 15, "verdicts": {"BARE_METAL": 8, "HARD_PUSH": 7},
 "files": 3, "loc": 87, "deps": 0, "concepts": 7}
```
17/17 green · tree clean · --final PASS · verdict trail: ITERATIONS.md.

## Final shapes
- **exam.json** — ONE data file: {path, task (bare list of outputs), scoring
  {"match": "exact"}, samples [{input, expected}], systems {kind: keyword,
  any, then, else}}. Declared lenses, nothing inferred, unknown keys loud.
- **nb/engine.py** — seven concepts: load, locate, invoke, run, as_loss,
  main (CLI), _check. Pure functions + loud validation. loc=87, stdlib-only.
- **CLI**: `python -m nb <bench_root> /sentiment artifact.json` → artifact
  {path, systems: {name: {cases: [{id, input, want, got, score}], score}}}.

## Best discovery
The **two-tier interpret-alone law**: a return value can lean on its caller;
a file on disk must carry its own identity. Same law kept per-case input
(c11) and restored the CLI artifact's path (c15). Companion laws: probe ≠
entry (c9), argument entry ≠ data entry (c12), reading ≠ checking (c14),
declared beats inferred (c1), index-as-id (c10).

## Most expensive mistake
c1: deriving the answer space from samples. Inference absorbs typos as
classes, shrinks the answer space silently, smuggles scoring into code —
one full cycle to falsify, and it shaped the tree's honesty: declare,
then loud-check.

## Angle status
Partially falsified, honestly. Maximal inference died in c1. Surviving
datacentric metal: ONE data file, declared answer-space + scoring, systems-
as-data (one spec kind), pure functions + loud checks, probe ≠ entry,
index-as-id, two-tier interpret-alone.

## To the other nine searchers
Write the invariant test before you push (c15 was pinned test-first).
Escalate in the same cycle when a probe comes up empty. Never delete a
schema check because "the reads make it redundant".

## Essence
*A benchmark is one JSON exam paper that declares its own answer space and
grading policy, plus seven tiny loud functions — nothing inferred, nothing
silent, a file must interpret alone.*