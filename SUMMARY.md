# SUMMARY — s07 datacentric (search/s07, final)

## Final golem report (verbatim)
```json
{"cycles": 15, "verdicts": {"BARE_METAL": 8, "HARD_PUSH": 7},
 "files": 3, "loc": 87, "deps": 0, "concepts": 7}
```
17/17 tests green. Tree clean. --final PASS.

## Final shapes
- **exam.json** — ONE data file: {path, task (bare list of outputs),
  scoring {"match": "exact"}, samples [{input, expected}],
  systems {name: {kind: keyword, any, then, else}}}. Declared lenses,
  nothing inferred, unknown keys loud at load.
- **nb/engine.py** — seven concepts: load, locate, invoke, run, as_loss,
  main (CLI), _check. Pure functions + loud validation. loc=87, stdlib-only.
- **CLI**: `python -m nb <bench_root> /sentiment artifact.json` → artifact
  {path, systems: {name: {cases: [{id, input, want, got, score}], score}}}.

## Verdict trail
c1 BM (inference is a cleverness engine — declared lenses are metal) ·
c2 HP (_check_exam fused into load) · c3 BM (CLI — runnable by a person) ·
c4 HP (context deleted) · c5 BM (as_loss — vision contract) ·
c6 HP (grade fused into run) · c7 HP (const = keyword with any=[]) ·
c8 HP (_check required param; kind escalation broke 7 tests) ·
c9 BM (probe ≠ entry; run's duplicate gate) · c10 HP (sample id — index IS
the id) · c11 BM (artifact echoes — per-case input must interpret alone) ·
c12 BM (invoke's gate — argument entry ≠ data entry) ·
c13 HP (SCORE_KEYS inlined — the one policy is the literal) ·
c14 BM (schema check — reading ≠ checking) ·
c15 BM (CLI artifact path — a file on disk outlives the invocation).

## Best discovery
The **two-tier interpret-alone law**: a return value can lean on its caller;
a file on disk must carry its own identity. Same law kept per-case input
(c11) and restored the CLI artifact's path (c15) — one principle, two tiers.
Companion laws: probe ≠ entry (c9), argument entry ≠ data entry (c12),
reading ≠ checking (c14), declared beats inferred (c1), index-as-id (c10).

## Most expensive mistake
c1: deriving the answer space from samples. Inference absorbs typos as
classes, shrinks the answer space silently, and smuggles scoring into code.
One full cycle to falsify — and it shaped the whole tree's honesty: declare,
then loud-check.

## Angle status
Partially falsified, honestly. Maximal inference died in c1. Surviving
datacentric metal: ONE data file, declared answer-space + scoring policy,
systems-as-data (one spec kind), pure functions + loud checks, probe ≠
entry, index-as-id, two-tier interpret-alone.

## To the other nine searchers
Write the invariant test before you push (c15 was pinned test-first).
Escalate in the same cycle when a probe comes up empty. Never delete a
schema check because "the reads make it redundant" — an unknown key riding
along silently is drift's front door. And a benchmark is one exam paper that
declares its own answer space: nothing inferred, nothing silent.

## Essence
*A benchmark is one JSON exam paper that declares its own answer space and
grading policy, plus seven tiny loud functions — nothing inferred, nothing
silent, a file must interpret alone.*