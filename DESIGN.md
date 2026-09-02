# s06 — protocol / contracts-first

Hypothesis (search/ANGLES.md): the product is the PROTOCOLS — Task, Scorer,
System, Data — plus the records that flow between them; implementations are
trivial one-liners.  Bare metal if swapping any implementation changes zero
lines outside it and the public surface stays ≤ ~10 public types.  Falsified
if protocols multiply beyond ~10.

**Interim verdict (cycle 3): the angle is being partially falsified by the
metal.**  Named protocols ARE noise when the convention carries the
contract: Scorer (deleted as a type — it's `(case, prediction) -> float`,
a convention), Scored (deleted — the artifact page is `{**case, prediction,
score}`, self-describing JSON), Task (deleted from the engine — the task
pillar lives in benchmark.json, which the engine passes through).  What
survives is ONE runtime-checkable protocol (System) + ONE record (Case) +
ONE interpreter (Exam).  The contract layer shrank from 6 named types to 2.

## shape

    benchmark.json      the exam as pure data: path (ontology), task (in/out
                        schema — pass-through, the engine does not read it),
                        cases, scoring kind
    Exam.run(system)    -> graded artifact (per-case scores + aggregate + loss)
    Exam.as_loss()      -> (system) -> float, lower is better
    System.invoke(x)    -> prediction   (structural conformance, no inheritance)
    scorer(case, pred)  -> float        (any callable; data-declared kind OR
                        injected python — the swap seam)

## concept table

| concept | pillar | why undeletable (so far) | survived |
|---|---|---|---|
| Case | DATA | one exam page (input, expected); the unit flowing through run; the only record the engine reads | 1 |
| System | SYSTEM | the exam taker: invoke(input)->prediction; structural conformance IS the AI-API contract; runtime-checkable so `isinstance` tells friend from stranger | 1 |
| exact_match | SCORING | the dumbest scorer; data-declared scoring needs at least one builtin kind | 0 |
| Exam | ALL | benchmark = data+scoring; system is the argument (run / as_loss — two vision invariants) | 1 |
| load | DATA | a benchmark is data; the only directory reader | 0 |
| locate | DATA | ontology path -> exam (vision invariant /<task?>/<domain?>/<language?>) | 0 |
| main | UX | `python -m nb <bench> <system>`: runnable without pytest archaeology | 0 |

Deleted (noise — protocols that only had annotation-work):
- Scorer protocol → class (cycle 1), Scorer type alias (cycle 2): the
  convention `(case, prediction) -> float` carries the contract.
- Scored record (cycle 3): `{**case, prediction, score}` is self-describing.
- Task record + Exam.task + artifact `conforms` (cycle 3): the task pillar
  is data (benchmark.json), not engine state; `conforms` was a second
  verdict competing with `score` (two sources of truth).

Current: 7 concepts, 4 files, 77 loc.  Public surface: Case (TypedDict),
System (Protocol), exact_match, SCORINGS, Exam, load, locate, main = 8 names.
Cycle 4+ will attack System-as-Protocol and locate.