# SUMMARY — s03 compiler / AI-API

## final shapes

```
engine (nb/, 5 files, 207 loc, 0 deps, stdlib only, ZERO classes):
  scoring.py   score(spec, pred, expected) -> 0..1        exact|partial|weighted
  benchmark.py run(bench, system) -> artifact             bench = (task, scoring, cases)
               as_loss(bench) -> (System) -> float       BARE_METAL (cycle 15)
  system.py    compile_system(spec) -> callable           THE AI-API: invoke(input)->pred
               backends: stub (absorbs const), http, chain, agent
  load.py      load(path) -> (task, scoring, cases)       + main() CLI
  __init__.py  exports: score, compile_system, run, as_loss

benchmark = a directory (pure data, never Python):
  task.json + scoring.json + cases.json + systems/*.json
  /<task?>/<domain?>/<language?> ontology
```

## golem report (verbatim)

{
  "cycles": 15,
  "verdicts": {
    "HARD_PUSH": 8,
    "NOISE_REMOVED": 6,
    "BARE_METAL": 1
  },
  "files": 5,
  "loc": 207,
  "deps": 0,
  "concepts": 10
}

## best discovery

The cycle-6 agent probe: a full tool-loop agent (model + tools +
budget + controller loop) absorbed as ONE backend entry with zero
core changes — the strongest possible confirmation that "every AI
system is a spec compiled to one callable." Then cycle 13's mini-
version: `const == stub with rules:{}` — the dumbest system and the
keyword system are ONE concept. Convergence keeps happening at the
spec level, never at the core level.

## most expensive mistake

The classes. Task, Case, Exam, Scoring, Benchmark — five classes,
five cycles (9,10,11,15 + Case in 8) to prove all of them were
wrappers around data. If I had started data-first (the "benchmark is
three JSON files" insight) the search would have been ~4 cycles
shorter. Classes are where redesigns hide their noise.

## advice for the other nine searchers

1. **stub/const/http/chain/agent transfer to every angle** — they are
   pure data specs compiled to `invoke(input)->pred`. Whatever your
   angle's core looks like, the system side can be EXACTLY this: a
   `kind`-keyed dict → callable table. A chain's steps are specs; an
   agent's tools are specs. Composition never needs core concepts.
2. **Kill your classes early and test the wreck.** Every class in this
   engine died leaving only data + a free function. The tests that
   "broke" were referencing names, not behavior.
3. **A constant system is a stub with empty rules.** Don't ship two
   backends for one idea.
4. **The CLI's only real error is an unknown name.** Empty systems
   dir = a quiet zero-exam; a graded exam is never an error (exit 0
   with score 0.5 is DATA).
5. **Keep `as_loss` at 3 lines over `run`.** It is the vision's bridge
   to prompt optimizers; the golem law itself defends it.
6. **empty-rules stub makes your acceptance bar self-contained**: the
   dumb system needs no special backend — it proves scoring
   discriminates using the same concept as the good system.