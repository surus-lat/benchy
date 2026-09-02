# s06 — protocol / contracts-first

Hypothesis (search/ANGLES.md): the product is the PROTOCOLS — Task, Scorer,
System, Data — plus the records that flow between them; implementations are
trivial one-liners.  Bare metal if swapping any implementation changes zero
lines outside it and the public surface stays ≤ ~10 public types.  Falsified
if protocols multiply beyond ~10.

**Interim verdict (cycle 6): the angle is falsified in its strong form,
confirmed in its weak form.**  Every NAMED contract died as annotation-cargo:
Scorer, Scored, Task, System (Protocol), Case (TypedDict), SCORINGS (registry).
What survives is the contracts as CONVENTIONS carried by the values:
scorer(case, pred)->float, system.invoke(x)->pred, case {input, expected},
artifact {**case, prediction, score}.  Swapping any implementation still
changes zero lines outside it — the seam is duck-typing on the convention,
which is what contracts-first actually bought.  Public surface: 5 module
names, not 10 types.

## shape

    benchmark.json      the exam as pure data: path (ontology), task (in/out
                        schema — pass-through, the engine does not read it),
                        cases
    Exam.run(system)    -> graded artifact (per-case scores + aggregate + loss)
    Exam.as_loss()      -> (system) -> float, lower is better
    System.invoke(x)    -> prediction   (structural conformance, no inheritance)
    scorer(case, pred)  -> float        (any callable, injected at Exam
                        construction — the swap seam; exact_match is the
                        built-in default; data-declared kinds deleted cycle 6)

## concept table

| concept | pillar | why undeletable (so far) | survived |
|---|---|---|---|
| exact_match | SCORING | the dumbest scorer; the built-in default | 0 |
| Exam | ALL | benchmark = data+scoring; system is the argument (run / as_loss — two vision invariants) | 2 |
| load | DATA | a benchmark is data; the only directory reader | 0 |
| locate | DATA | ontology path -> exam (vision invariant /<task?>/<domain?>/<language?>) | 1 |
| main | UX | `python -m nb <bench> <system>`: runnable without pytest archaeology | 0 |

## bare metal (survived a deletion attempt)

- locate (cycle 4): the ontology path is the vision's addressing scheme
  (/<task?>/<domain?>/<language?>), not a directory convention.
- as_loss (cycle 8): VISION/IDEAS name the loss export the headline feature —
  "export the benchmark as a new loss function for prompt-optimizers" — and
  the acceptance bar demands `as_loss()` ranks the stubs. It is the identity
  of the benchmark-as-loss view (angle s01's whole thesis), not a feature.
  The deletion broke the acceptance test itself; a benchmark that cannot be
  handed to an optimizer is not benchy.

## deleted (noise — protocols that only had annotation-work)

- Scorer protocol class (cycle 1), Scorer type alias (cycle 2): the
  convention `(case, prediction) -> float` carries the contract.
- Scored record (cycle 3): `{**case, prediction, score}` is self-describing.
- Task record + Exam.task + artifact `conforms` (cycle 3): the task pillar is
  data (benchmark.json), not engine state; `conforms` was a second verdict
  competing with `score`.
- System Protocol (cycle 5): structural conformance IS the contract; the
  runtime-checkable type was annotation-cargo.
- Case TypedDict (cycle 5): `{input, expected}` lives in the values.
- All type annotations in core.py (cycle 5): core.py became a pure
  conventions docstring — the honest endpoint of contracts-first.
- SCORINGS registry (cycle 6): a registry of one entry is a fake choice;
  load hardcodes exact_match, custom scoring rides the injected-scorer seam.
- scoring-kind-in-data (cycle 7): benchmark.json's "scoring": {"kind": ...}
  became a dangling pointer once the registry died — data that looked like
  a choice but selected nothing. Custom scoring is injected python, not
  data-declared kinds (the s05 finding, confirmed here).

Current: 5 concepts, 4 files, 69 loc.  Public surface (module names):
exact_match, Exam (+run, as_loss), load, locate, main = 5.