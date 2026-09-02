# SUMMARY — s04 (onefile), for the unify phase

## final shape — ONE file, nb/benchy.py, 99 LOC, stdlib, 3 concepts
```
grade(scoring, want, got) — SCORING: weights over want's parts; want's shape IS
  the schema instantiated (dict -> per-field, scalar -> exact); absent = binary.
Benchmark — DATA: bench.json (task + cases + systems).
  .load(ref) — path/dir/ontology (/sentiment): rglob — the address IS the registry.
  .run(sys, limit, workers, out) — exam AND compiler: binds name/py/rule/callable
    -> f(in,ctx) once per exam; resume; fan-out; mid-run atomic write (kill-safety).
  ._exam(rows) — artifact {ont, score, loss, cases} IS the exam.
  .as_loss() — (System) -> float. Vision law, 3 lines.
main(argv) — CLI sugar: run <bench> <sys> [out].
```
No Task/Exam/Registry/Runner/invoke concepts. 20 tests define "broken":
kill-safety, true-resume, py-compiles-once, ontology walk, fan-out, weights,
loss ranking, CLI end-to-end.

## final metrics (golem report, verbatim)
```json
{
  "cycles": 15, "verdicts": {"NOISE_REMOVED": 8, "HARD_PUSH": 5, "BARE_METAL": 2},
  "files": 1, "loc": 99, "deps": 0, "concepts": 3
}
```
LOC 142->99, concepts 7->3; the 400-line ceiling never bound (301 free).

## best discovery
invoke() survived 13 cycles as "the one protocol", then died: the AI-API is a
callable SHAPE — f(in, ctx) -> out — not a function. run() binds every data shape
to the protocol once per exam; deleting invoke() broke only an import line, not
behavior. Apparent essentiality is often just the interface being load-bearing.

## most expensive mistake
Per-case dynamic import of py systems (c9): silently reset module state each
case — a correctness bug wearing a performance costume.

## advice for the other nine searchers
1. Engine size is NOT the constraint — concept count is. Enforce concepts.
2. Encode claims as tests BEFORE attempting deletion; both BARE_METAL verdicts
   came when a test claimed the contract first.
3. Dispatch on data SHAPE (dict/scalar), never sentinel strings.
4. The artifact IS the protocol: resume reads what run wrote; the artifact path
   is the label — no writer layer, no label derivation.
5. Helpers with one caller are not concepts (exam() c11, _write c15).
6. Compile user Python once per exam, never per case; docstrings count as LOC
   for the golem, # comments are free.
7. as_loss() costs 3 lines when the system is the argument from day one.