# s06 — protocol / contracts-first (SUMMARY for unify)

## Final metrics (golem report verbatim)

```json
{
  "cycles": 15,
  "verdicts": {
    "NOISE_REMOVED": 2,
    "HARD_PUSH": 9,
    "BARE_METAL": 4
  },
  "files": 2,
  "loc": 46,
  "deps": 0,
  "concepts": 3
}
```

`golem.py check --final` → PASS (15 cycles ≥ 12, last verdict HARD_PUSH).

## Final shape

- 46 loc, 3 concepts, 2 files, 0 deps — `nb/exam.py` (all behavior) + `nb/__main__.py` (2-line shim)
- Public surface: 3 — `Exam` (+run, as_loss), `locate`, `main`
- Engine: `python -m nb <root> <ontology_path> <system>` — offline, no pytest archaeology
- Artifact: `{benchmark, cases[{input,expected,prediction,score}], score}` — pure
  facts; `loss` is a lens (1−score) derived at each consumer's seam
- Benchmark: one tracked data file (`bench/hello/benchmark.json`) located by
  ontology path `/hello` — pure data, never required Python

## Angle status

Partial-falsified, sharpened twice in the final stretch. Every named type died
as annotation-cargo (Scorer, Scored, Task, System Protocol, Case, SCORINGS,
Callable — and c14: the named *default* exact_match too; unnamed defaults are
more honest). The class-shaped surface `Exam.run`/`Exam.as_loss` remained
load-bearing metal to the end. **Duck-typing governs the seams; named method
syntax carries the invariants; named defaults are cargo too.**

## Best discovery

The Exam-class BARE_METAL (c12): dissolving to s03-style free functions broke 7
tests AND grew concepts 4→5. Two metal facts: the class is the
concept-COMPRESSOR (methods count free inside it), and the vision invariant is
written in method syntax (`benchmark.run(system)`, `benchmark.as_loss()`).
**Direct evidence for the unify on s01-closure vs s03-free-functions: the
closure/class surface wins — a tuple can't carry a spec.**

## BARE_METAL badges (4)

1. `locate` — ontology addressing is a vision invariant (c4)
2. `as_loss` — benchmark-as-loss is the vision's identity, not a feature (c7)
3. `main` — offline end-to-end without pytest is the acceptance bar (c10)
4. `Exam` class — the public surface IS the spec (c12)

## Most expensive mistake

Half the search mistook annotation for contract: `core.py` lingered 8 cycles
as a conventions docstring before c13 proved the values + tests already carry
the spec (a spec written in three places is two places of drift risk).

## Advice for the other nine searchers

- Delete named defaults — the seam is the constructor parameter, not a name.
- Delete write-only artifact fields — if only one consumer reads it, let that
  consumer derive it at its own seam.
- Keep a `main` — an engine only reachable via pytest is archaeology.
- Verify your benchmark data is actually git-tracked (root `*.json` gitignore
  silently untracked whole exams in three trees: s05, s06, and s04's state).

## One-line essence

The exam is pure data located by ontology path, graded through one class-shaped
surface (`Exam.run(system)`, `Exam.as_loss()`) over two duck-typed seams
(`.invoke`, injected scorer) — everything named at the seams was
annotation-cargo, everything on the public noun was the spec.