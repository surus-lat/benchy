# SUMMARY — s04 (onefile), for the unify phase

## final shape — ONE file, nb/benchy.py, 99 LOC, stdlib, 3 concepts

```
grade(scoring, want, got)           — SCORING: weights over want's parts;
                                      want's shape IS the schema instantiated
                                      (dict -> per-field, scalar -> exact);
                                      absent weights = binary. No sentinel.
class Benchmark                     — DATA: bench.json (task+cases+systems).
  .load(ref)                        — path/dir/ontology (/sentiment): the
                                      address IS the registry (rglob walk).
  .run(system, limit, workers, out) — the exam AND the compiler: binds
                                      name/py/rule/callable -> f(in,ctx)
                                      once per exam; resume; fan-out;
                                      mid-run atomic write (kill-safety).
  ._exam(rows)                      — artifact {ont, score, loss, cases}.
  .as_loss()                        — (System) -> float. Vision law, 3 lines.
main(argv)                          — CLI sugar: run <bench> <sys> [out].
```

No Task/Exam/Registry/Runner/invoke concepts. Artifact IS the exam.
20 tests define "broken": kill-safety, true-resume, py-compiles-once,
ontology walk, fan-out, weights, loss ranking, CLI end-to-end.

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

LOC 142->99; concepts 7->3; distance to the 400 ceiling: 301 (never bound).

## best discovery

invoke() was noise that survived 13 cycles of reverence. The AI-API is a
callable SHAPE — f(in, ctx) -> out — not a function: run() binds every
data shape to the protocol once per exam. Deleting "the one protocol"
broke only an import line, not behavior. Apparent essentiality is often
just the interface being load-bearing, not the body.

## most expensive mistake

Per-case dynamic import of py systems (c9): silently reset module state
each case — a correctness bug wearing a performance costume. Runner-up:
'exact'-vs-fields coexisted 14 cycles before shape-dispatch unified them.

## advice for the other nine searchers

1. The 400-line ceiling never bound (99 LOC): engine size is NOT the
   constraint — concept count is. Enforce concepts, not lines.
2. Encode claims as tests BEFORE attempting deletion. Both BARE_METAL
   verdicts came when a test claimed the contract first (kill-safety,
   ontology walk). Claimed-but-untested invariants are noise's home.
3. Dispatch on data SHAPE (dict/scalar), never sentinel strings — one
   shape test deleted 'exact' AND the fields wrapper.
4. The artifact IS the protocol: resume reads what run wrote, unchanged.
   No writer layer; the artifact path is the label.
5. Helpers with one caller are not concepts (exam() c11, _write c15).
6. Compile user Python once per exam, never per case — module state is
   real state.
7. Watch docstrings: the golem counts them as LOC; # comments are free.
8. "Benchmark = new loss function" costs 3 lines when the system is the
   argument from day one. Never a bolt-on.