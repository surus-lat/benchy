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
which is what contracts-first actually bought.  Public surface: module
names (5 at cycle 6; 4 after cycle 9 fused load into locate), not 10 types.

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

The conventions (case {input, expected}, artifact shape, .invoke) are carried
by the VALUES and pinned by the tests — since cycle 13 there is no core.py:
a doc duplicating the spec the values+tests already state was noise.

Since cycle 14 the default scorer is an anonymous lambda inside locate
(exact_match deleted as a named public concept): no test or caller ever
needed the default BY NAME — the custom-scorer swap test injects
positionally, proving the seam is the constructor parameter, not a named
default. An unnamed default is more honest: the seam is the contract,
locate's lambda is just its default policy.

## concept table

| concept | pillar | why undeletable (so far) | survived |
|---|---|---|---|
| Exam | ALL | 3 survivals. cycle 12 dissolution broke 7 tests AND grew concepts 4->5: the class is the concept-COMPRESSOR (methods count free inside it) and carries the vision invariant's own syntax — `benchmark.run(system)`, `benchmark.as_loss()` (GOLEM.md law 6) | 3 |
| locate | DATA | ontology path -> exam (vision invariant /<task?>/<domain?>/<language?>); since cycle 9 the only constructor (load fused in); since cycle 14 also owns scoring policy: it injects the default scorer as an anonymous lambda | 2 |
| main | UX | cycle 10 deletion broke test_cli_runs_offline_end_to_end: the acceptance bar is "runs offline, end to end" WITHOUT pytest — an engine only reachable via pytest is archaeology, not a product. Lives in exam.py (fused); __main__.py is a 2-line shim | 1 |

## bare metal (survived a deletion attempt)

- locate (cycle 4): the ontology path is the vision's addressing scheme
  (/<task?>/<domain?>/<language?>), not a directory convention.
- as_loss (cycle 8): VISION/IDEAS name the loss export the headline feature —
  "export the benchmark as a new loss function for prompt-optimizers" — and
  the acceptance bar demands `as_loss()` ranks the stubs. It is the identity
  of the benchmark-as-loss view (angle s01's whole thesis), not a feature.
  The deletion broke the acceptance test itself; a benchmark that cannot be
  handed to an optimizer is not benchy.
- main (cycle 10): deleting the CLI broke test_cli_runs_offline_end_to_end.
  GOLEM.md's acceptance bar says the engine must "run this offline, end to
  end" — an engine whose only entry point is pytest is not runnable, it is
  archaeology. The behavior (main) is metal; the FILE was noise: fusing it
  into exam.py + a 2-line __main__ shim kept the concept and shed 6 loc of
  import ceremony and docstring duplication.
- artifact's `benchmark` identity field (cycle 11): deleting it broke two
  tests. The artifact is graded EVIDENCE written to disk — once on disk,
  nothing ties it to its exam except this field (filename/location identity
  evaporates on copy). An unlabeled grade is a mean without an exam. The
  old benchy's run_outcome.json carries run identity at top level for the
  same reason: report-side reading needs self-describing data.
- Exam, the class (cycle 12, 3rd survival): dissolving it into free
  functions over a (cases, scorer, path) tuple — s03's move — broke 7
  tests and the golem ITSELF growled GREW concepts 4->5. Two metal facts:
  (a) the class is the concept-compressor — `run` and `as_loss` count as
  free top-level defs when they escape it; (b) the vision invariant is
  written in method syntax (`result = benchmark.run(system)`,
  `loss = benchmark.as_loss()`, GOLEM.md law 6) — the surface IS the
  spec. A tuple can't carry a spec.

## deleted (noise — protocols that only had annotation-work)

- Scorer protocol class (cycle 1), Scorer type alias (cycle 2): the
  convention `(case, prediction) -> float` carries the contract.
- exact_match as a NAMED public concept (cycle 14): the default scorer is
  locate's private policy now — an anonymous lambda on the injection seam.
  Zero tests or callers referenced it by name (the swap test injects
  positionally); a named default invited `from nb.exam import exact_match`,
  which would have made the default itself a load-bearing public surface.
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
- load (cycle 9, fused into locate): a separate loader after locate reads
  the files it searched was a concept doing locate's job twice; the
  ontology path is the only address and locate the only constructor.
- core.py, the conventions-docstring file (cycle 13): after cycle 5 it had
  zero code — only a doc restating the conventions. Deleted whole: the
  conventions live in the values and are PINNED BY THE TESTS
  (test_artifact_is_graded_json pins the artifact schema; stubs.py documents
  .invoke). A spec written in three places is two places of drift risk.
  Engine is now 3 files: __init__ (docstring), __main__ (2-line shim),
  exam.py (all behavior).

Current: 3 concepts, 3 files, 48 loc.  Public surface (module names):
Exam (+run, as_loss), locate, main = 3.