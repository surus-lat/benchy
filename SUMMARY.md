# SUMMARY — s04 (onefile), for the unify phase

## final shape

ONE file, `nb/benchy.py`, 99 LOC, stdlib only, 3 concepts:

```
grade(scoring, want, got)          — SCORING: weights over want's parts;
                                     want's shape IS the output schema
                                     instantiated; absent weights = binary.
class Benchmark                    — DATA: task + cases + systems (data).
  .load(ref)                       — path/dir/ontology (/sentiment):
                                     the address IS the registry (rglob).
  .run(system, limit, workers, out)— the exam AND the compiler: binds
                                     name/py/rule/callable -> f(in,ctx)
                                     once per exam; resume; fan-out;
                                     mid-run atomic write (kill-safety).
  ._exam(rows)                     — rows -> artifact {ont, score, loss, cases}
  .as_loss()                       — (System) -> float, vision law.
main(argv)                         — CLI sugar: run <bench> <sys> [out].
```

Benchmark = bench.json (task + cases + optional scoring weights +
systems). No Task/Exam/Registry/Runner/invoke concepts. Artifact IS the
exam. 20 tests define "broken" (kill-safety, true-resume, py-compiles-
once, ontology walk, fan-out, weights, loss ranking).

## final metrics (golem report, verbatim)

```json
{
  "cycles": 15,
  "verdicts": {
    "NOISE_REMOVED": 8,
    "HARD_PUSH": 5,
    "BARE_METAL": 2
  },
  "files": 1,
  "loc": 99,
  "deps": 0,
  "concepts": 3
}
```

LOC trajectory: 142 (c1) -> 99 (c15). Concepts 7 -> 3.

## best discovery

invoke() was noise that survived 13 cycles of reverence. The AI-API is a
callable SHAPE — f(in, ctx) -> out — not a function: run() binds every
data shape (name, {"py": ...}, rule+default, callable) to the protocol
once per exam. Deleting "the one protocol" broke only an import-line
test, not behavior. The lesson generalizes: a concept's apparent
essentiality is often just its interface being load-bearing, not its body.

## most expensive mistake

Per-case dynamic import of py systems (c9): silently reset module state
every case — a correctness bug wearing a performance costume. It took a
dedicated stateful-system test (counter across cases) to expose. Second
place: the "exact"-vs-fields split in grade(), which cost 14 cycles of
coexistence before shape-dispatch unified it in 4 lines.

## what I would tell the other nine searchers

1. The size ceiling (400 lines) never bound — engine landed at 99 LOC.
   Engine size is NOT the constraint; concept count is. Measure and
   enforce concepts, not lines, and fusion does the rest.
2. Encode claims as tests BEFORE attempting deletion. The two BARE_METAL
   verdicts came exactly when a test claimed the contract first
   (kill-safety, ontology walk). Claimed-but-untested invariants are
   noise's favorite hiding place.
3. Dispatch on data SHAPE (dict/scalar), never on sentinel strings or
   wrapper keys — one shape test deleted 'exact' AND the fields wrapper.
4. The artifact IS the protocol: resume reads what run wrote, unchanged.
   No writer layer, no formatter, no label derivation — the path is the
   label.
5. Helpers with one caller are not concepts (exam() c11, _write c15).
   The onefile forcing function makes this visible; use it.
6. Compile user Python once per exam, never per case — module state is
   real state, and re-importing it is a bug.
7. The golem grows, not the engine: watch docstrings (they count as
   LOC); push prose into # comments.
8. The vision's "benchmark = new loss function" costs 3 lines when the
   system is the argument from day one. Do not make it a bolt-on.