# s06 — protocol / contracts-first

Hypothesis (search/ANGLES.md): the product is the PROTOCOLS — Task, Scorer,
System, Data — plus the records that flow between them; implementations are
trivial one-liners.  `nb/core.py` is pure contracts (no behavior); `nb/exam.py`
is the dumbest interpreter that satisfies them.  Bare metal if swapping any
implementation changes zero lines outside it and the surface stays ≤ ~10
public types.  Falsified if protocols multiply beyond ~10.

## shape

    benchmark.json      the exam as pure data: path (ontology), task (in/out
                        schema), cases, scoring
    Exam.run(system)    -> graded artifact (per-case scores + aggregate + loss)
    Exam.as_loss()      -> (system) -> float, lower is better
    System.invoke(x)    -> prediction   (structural conformance, no inheritance)
    Scorer(case, pred)  -> float        (structural; data-declared kind OR
                        injected python — the swap seam)

Ontology: a benchmark self-describes as "/sentiment"; locate() maps an
ontology path to the exam under a bench root.

## concept table

| concept | pillar | why undeletable (so far) | survived |
|---|---|---|---|
| Task | TASK | the program description; declares the output enum the artifact checks conformance against | 0 |
| Case | DATA | one exam page (input, expected) — the unit that flows through run | 0 |
| Scored | SCORING | the graded page — the artifact's atom; what the report side reads | 0 |
| System | SYSTEM | the exam taker: one method, invoke(input)->prediction; structural conformance IS the AI-API contract | 0 |
| Scorer | SCORING | what good means: (case, prediction)->float; the swap seam keeping implementation changes local | 0 |
| exact_match | SCORING | the dumbest scorer; data-declared scoring needs at least one builtin | 0 |
| Exam | ALL | benchmark = task+data+scoring; system is the argument (run / as_loss — two vision invariants) | 0 |
| load | DATA | a benchmark is data; this is the only directory reader | 0 |
| locate | DATA | ontology path -> exam (vision invariant /<task?>/<domain?>/<language?>) | 0 |
| main | UX | `python -m nb <bench> <system>`: runnable without pytest archaeology | 0 |

Baseline: 10 concepts, 6 public types, 4 files, ~99 loc.  Every cycle tries
to delete one; the table keeps only what survives.