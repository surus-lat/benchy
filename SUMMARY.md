# SUMMARY — s01 loss-first, final state for unify

## final shape

```
loss = load("/sentiment")     # the benchmark IS a loss: (System) -> float
loss(system)                  # one evaluation -> float (lower = better)
loss.trace                    # the receipt: {score, cases:[{in,want,got,score}]}
```

- Engine: nb/bench.py — ONE function (`load`), 34 loc, stdlib-only (json,
  pathlib), zero deps. It globs bench/**/bench.json, checks the scoring key
  loudly, returns the per-load loss closure.
- Benchmark = data: bench/hello/bench.json = {path, scoring, cases} — no
  Python in the benchmark. Systems are importable `solve` callables
  (bench/hello/systems/{good,dumb}.py); the engine has NO system loader.
- Artifact: json.dumps(loss.trace) — self-contained per-case evidence
  (in/want/got/score) + aggregate.
- Tests: nb_tests/test_engine.py, 12 tests green, offline.

## metrics (golem report, verbatim)

```json
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
```

`golem.py check --final` → PASS (15 cycles, last verdict BARE_METAL).

## best discovery

The entire exam/report side is a PROJECTION of the loss view — it needed
zero new concepts. `loss.trace` IS the graded artifact; the exam is the
cases it iterates; the report is its per-case verdicts. Everything the
old design spread across Bench.run / as_loss / artifact builder / system
loader collapsed into one 34-line closure. The angle's falsification
condition ("exam/report needs concepts the loss can't generate") never
fired once in 15 cycles.

## most expensive mistake

Trusting green tests as proof of noise. Cycle 11 (scoring key) and cycle 15
(float cast): deletion kept 12/12 green, which read as "noise" — but both
were unguarded guards, not noise. The honest protocol that fixed it:
when a deletion stays green but judgment says metal, write the guard test
FIRST, delete again, and let the badge be earned by a real failure. Two of
six BARE_METAL badges only exist because the unguarded-guard pattern was
recognized the second time.

## advice for the other nine searchers

1. Prove your shape can't be hoisted: per-instance receipt state (two
   benchmarks must not clobber) and a pure `(System) -> float` signature are
   forced constraints — cycles 12/13 both bounced on them.
2. Artifact must be interpretable ALONE (want in trace): optimizers will
   read your JSON with no access to the source data. "Derivable from
   bench.json" is not a defense.
3. Guard your guards: for every loud-failure path (unknown scoring,
   verdict types), keep a test that FAILS when the guard is deleted.
   Green-on-delete means noise OR unguarded guard — judgment decides.
4. Loudness can be free: empty-cases ZeroDivisionError needed zero code.
   Before adding a check, check whether construction is already loud.
5. Direction is semantics: return loss (lower=better), not score. An
   optimizer must minimize your callable without knowing your conventions.
6. Keep the queue honest: cycles 13-15 were all queued candidates that
   BOUNCED. The engine that survives its own queue is the metal.

## the caveat for unify

Cases carry the task only by example — a task-only description with zero
cases can't be expressed (cycle 7). If the CREATE flow needs task-before-
data, unify must decide where the description lives; re-adding an engine-
unread key would be noise, but a convention-level answer (comment field,
README) is open.