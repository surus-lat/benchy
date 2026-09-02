# SUMMARY — s01 loss-first, final state for unify

## final shape

    loss = load("/sentiment")   # the benchmark IS a loss: (System) -> float
    loss(system)                # one evaluation -> float (lower = better)
    loss.trace                  # {score, cases:[{in, want, got, score}]}

- Engine: nb/bench.py — ONE function (`load`), 34 loc, stdlib-only, 0 deps.
- Benchmark = data: bench/hello/bench.json = {path, scoring, cases}. Systems
  are importable `solve` callables; the engine has NO system loader.
- Artifact: json.dumps(loss.trace) — self-contained per-case evidence.
- Tests: nb_tests/, 12 green, offline. `golem.py check --final` → PASS.

## metrics (golem report, verbatim)

{
  "cycles": 15,
  "verdicts": {
    "NOISE_REMOVED": 4,
    "HARD_PUSH": 5,
    "BARE_METAL": 6
  },
  "files": 1,
  "loc": 34,
  "deps": 0,
  "concepts": 1
}

## best discovery

The entire exam/report side is a PROJECTION of the loss view — zero new
concepts needed. loss.trace IS the graded artifact; the exam is the cases it
iterates; the report is its per-case verdicts. The angle's falsification
condition ("exam/report needs concepts the loss can't generate") never fired
in 15 cycles.

## most expensive mistake

Trusting green tests as proof of noise. Cycles 11 (scoring key) and 15
(float cast): deletion kept 12/12 green, which read as noise — both were
UNGUARDED GUARDS, not noise. Fix: when a deletion stays green but judgment
says metal, write the guard test FIRST, delete again, let the badge be earned
by a real failure.

## advice for the other nine searchers

1. Prove your shape can't be hoisted: per-instance receipt state + pure
   (System)->float signature are forced constraints (cycles 12/13 bounced).
2. Artifact must be interpretable ALONE (want in trace): "derivable from
   bench.json" is not a defense — optimizers read the JSON with no source.
3. Guard your guards: keep a test that FAILS when each honesty guard is
   deleted. Green-on-delete = noise OR unguarded guard; judgment decides.
4. Loudness can be free: empty-cases ZeroDivisionError needed zero code.
   Check whether construction is already loud before adding a check.
5. Direction is semantics: return loss (lower=better), not score.
6. Keep the queue honest: cycles 13-15 were queued candidates that BOUNCED.

## caveat for unify

Cases carry the task only by example — a task-only description with zero
cases can't be expressed (cycle 7). If CREATE needs task-before-data, unify
must decide where the description lives; an engine-unread key would be noise.