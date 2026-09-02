# s06 LEARNINGS — what the metal actually is under contracts-first

15 cycles. 10 concepts -> 3 (Exam, locate, main). 95 loc -> 46. 5 files -> 2.
Zero deps, stdlib only, 9/9 acceptance tests green throughout. The angle's
hypothesis (the product is the PROTOCOLS: named Task/Scorer/System/Data
types) is falsified in its strong form, confirmed in a sharper weak form.

## the honest falsification story

Every NAMED contract died as annotation-cargo — the pattern held without a
single exception across 15 cycles: Scorer protocol class (c1), Scorer type
alias (c2), Scored TypedDict (c3), Task record + conforms field (c3), System
Protocol (c5), Case TypedDict (c5), every remaining type annotation (c5),
SCORINGS registry (c6), data-declared scoring kind (c7), Callable on as_loss
(c11), and finally the conventions-docstring file core.py itself (c13) —
contracts-first's own endpoint: a file of pure types with zero behavior is a
doc, and a doc duplicating the spec the values+tests already carry is noise.

But the deepest rehabilitation of the angle arrived at cycle 12: the
CLASS-SHAPED SURFACE Exam.run(system) / Exam.as_loss() survived dissolution
where every flat annotation died. Two metal facts: (a) the class is the
concept-COMPRESSOR — methods count free inside it, so dissolving it GREW
concepts 4->5 (the golem itself growled); (b) the vision invariant is
written in method syntax (`result = benchmark.run(system)`,
`loss = benchmark.as_loss()` — GOLEM.md law 6). The public surface IS the
spec; a (cases, scorer, path) tuple cannot carry a spec.

So the verdict on contracts-first: DUCK-TYPING GOVERNS THE SEAMS; NAMED
METHOD SYNTAX CARRIES THE INVARIANTS. Named types at the seams were cargo;
named methods on the one public noun were the product.

## pillar by pillar

### TASK (the program description)
The metal is a FIELD IN THE DATA, not engine state. benchmark.json carries
`task: {input, output}` and the engine NEVER READS IT — it is the
declaration the exam-taker (compiler/system side) compiles against, a
pass-through contract in pure data. Task record, Case TypedDict and the
artifact's `conforms` verdict all died (c3/c5): a second verdict field was
the engine re-deriving what scoring already says, and a typed Case was the
schema {input, expected} written twice. The task pillar costs ZERO engine
concepts.

### SCORING
The metal is an INJECTED CALLABLE at one seam: `scorer(case, prediction)
-> float`, injected at Exam construction. What died around it: the protocol
class, the type alias, the registry (a registry of one entry is a fake
choice — c6), data-declared scoring kinds (a dangling pointer once the
registry died — c7, confirming s05's finding: custom scoring is injected
python, never data-named kinds), and the NAMED default `exact_match` (c14)
— no test or caller ever referenced the default by name; the swap test
injects positionally, proving the seam is the constructor parameter, not a
named function. The default is now an anonymous lambda inside locate:
unnamed defaults are more honest than named ones, which invite
`from nb.exam import exact_match` and quietly become load-bearing surface.
In the artifact, per-case `score` + the mean are the facts; `loss` was a
write-only LENS over score (1-score) and died in c15 — consumers (as_loss,
main's print) derive the lens at their own seam.

### DATA (the exam)
benchmark.json IS the whole exam — path (ontology), task (pass-through
declaration), cases ({input, expected} values, no type) — and `locate` is
the ONLY constructor: ontology path `/<task?>/<domain?>/<language?>` ->
exam (BARE_METAL c4 — the vision's addressing scheme, not a directory
convention). locate's search-then-load split died in c9: a separate loader
after locate had read the files it searched was one concept doing its job
twice. Cycle 15's salvage finding lives here and belongs to every searcher:
the root .gitignore's `*.json` had silently UNTRACKED the entire exam —
"benchmark is DATA" means the data must be in version control or the engine
has no benchmark at all; a fresh checkout would have failed the acceptance
bar with LookupError. Fixed via `!bench/**/benchmark.json`.

### SYSTEM (the exam taker / compiler)
The metal is ONE STRUCTURAL CONVENTION: `system.invoke(input) -> prediction`.
The System Protocol died as annotation-cargo (c5) — structural conformance
IS the contract, and the test suite pins it with a class defined inline
(test_any_invoked_program_takes_the_exam). The system is the ARGUMENT, never
constructor state (IDEAS.md's own code idea): `exam.run(system)`,
`loss = exam.as_loss()(system)` — one exam, many takers, is the native path.
Per the 2026-09-01 steering: the first real taker is CLOUD (a foundational
model behind an AI-API); no serving machinery belongs in the engine. The
engine's entire system pillar is the duck-typed `.invoke` seam the cloud
adapter will satisfy.

## the four BARE_METAL badges

1. `locate` (c4) — ontology path -> exam: the vision's addressing scheme.
2. `as_loss` (c8) — the loss export is the IDENTITY (VISION/IDEAS call it
   the headline feature; the acceptance bar demands it ranks the stubs).
   A benchmark that cannot be handed to an optimizer is not benchy.
3. `main` (c10) — "runs offline, end to end" means WITHOUT pytest: an
   engine only reachable via pytest is archaeology, not a product. The
   behavior is metal; the FILE was noise (fused into exam.py + 2-line
   `__main__` shim, net -6 loc).
4. `Exam`, the class (c12, 3 survivals) — the concept-compressor carrying
   the vision invariant's own method syntax. Dissolution broke 7 tests AND
   grew concepts 4->5. The strongest rehabilitation of contracts-first's
   core claim in the whole tree.

(Plus the artifact's `benchmark` identity field, c11: evidence written to
disk must be self-describing — once on disk, nothing ties a grade to its
exam except that field. The old benchy's run_outcome.json carries run
identity at top level for the same reason.)

## what contracts-first bought, and what it cost

BOUGHT: the injection seams. Swapping scorer or system changes ZERO engine
lines — the angle's real product claim survived as two duck-typed seams
(scorer param, .invoke convention) plus the artifact's self-describing
shape. Every test of the engine is a test of a seam.

COST: ten concepts of named ceremony had to die before the shape appeared,
and the type-file instinct left a residue — core.py spent eight cycles as
a pure conventions docstring: the spec written in three places (values,
tests, doc) was two places of drift risk until c13 deleted it. The other
cost is epistemic: for half the search the angle mistook ANNOTATION for
CONTRACT. The golem's survived-attempt bar was the only thing that exposed
the difference — nothing named ever survived a deletion attempt except
`Exam.run`/`as_loss` method syntax itself.