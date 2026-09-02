# nb — the four pillars as data, the engine as five functions

**Angle s10 (salvage)**, cycles 1–12 built from zero. This session: the build.
The archaeology phase (cycles ~13–15) audits the old system read-only and may
donate ONLY what survives a deletion attempt inside THIS design.

## the shape

The four pillars of the ontology are DATA. The engine is four PURE functions
over that data — values in, values out, no classes, no registries, no I/O.

```
DATA     bench/hello/bundle/           one self-contained benchmark bundle:
           sentiment.json               the benchmark: task + scoring + cases
           systems.json                 system SPECS (exam-takers, as data)
ENGINE   nb/engine.py                  compile / grade / run / as_loss (pure)
         nb/__main__.py                python -m nb — the file layer + CLI
```

- **TASK pillar** lives in the benchmark data: `task.input` (a string text),
  `task.output` (declared choices) — and it is LOAD-BEARING (c8): grade
  refuses any exam whose answer key falls outside the declared choices.
  "takes a text, returns {pos,neg}" is data, enforced by the engine.
- **SCORING pillar** lives in the benchmark data: `scoring.rule` (match),
  `scoring.aggregate` (mean). Grading is a lookup, not a framework — and
  c11 fused the two scoring refusals into one literal: the declared scoring
  must be EXACTLY `{rule: match, aggregate: mean}` or grade refuses.
- **DATA pillar** is the cases list — the exam itself. n cases, each a
  (input, expected) pair; a row's POSITION in the artifact IS its id (c9
  deleted the explicit `case` index — derivable noise). Everything is a
  distribution; the artifact reports the point estimate plus per-case
  evidence.
- **SYSTEM pillar** is a SPEC compiled by the engine: `compile(spec)` turns
  data into `invoke(text) -> prediction`. Today: keyword + constant (offline,
  no network). Cloud kinds join as new spec kinds — the exam-taker is
  cloud-first per the steering addendum, and serving is long-term work, so
  the engine grows only at `compile`.

Vision invariants held: `run(benchmark, system)` — the system is the
ARGUMENT; `as_loss(benchmark)` exports the benchmark as a new loss function
over systems; the ontology path `/sentiment` locates the benchmark.

## concept table

| concept | pillar | why it cannot be deleted | survived |
|---|---|---|---|
| compile | system | the SYSTEM pillar: turns a spec into invoke(text)->pred; the only place the engine may grow (cloud kinds). c13 deleted silent case-folding magic — matching is LITERAL, the spec carries case. c14 pinned `default`: a constant system is the degenerate keyword (pos=[], default=pos) and the choice lives in the SPEC as data — hardcoding "neg" scores identically on a balanced exam (score-blind), only the behavior pin catches it. | 1 |
| grade | scoring+seam | the seam where ANY callable takes the exam — real APIs, workflows, cached runs bypass compile. c3 fused it into run and the seam test broke. c8 made it the TASK pillar's enforcement point: it refuses exam keys outside task.output.choices — the declaration is load-bearing. c11 pinned the aggregate refusal and fused the two same-kind refusals into one literal: `scoring != {rule: match, aggregate: mean}` is refused — exactly the scoring implemented, nothing declared-but-unread. c16 (DONATED from the old benchy run-loop contract, .staging/benchy/benchmark.py): a system failure on one case is EVIDENCE, never an abort — grade catches Exception per case, rows carry `prediction=None, error="Type: msg"`, the fused match scoring scores it 0 (None never equals a declared choice). Divergence from the old system, on purpose: it excluded errored samples from the aggregate; here a failed case scores 0 and stays IN — reliability lands in the one scalar the optimizer consumes. Probe-deletion broke 14 tests. c17 (DONATED from the old status vocabulary, src/outcome.py no_samples / AGENTS.md counts.no_samples_tasks): an empty exam is a broken exam, refused upfront — with the refusal deleted, as_loss crashed the optimizer with ZeroDivisionError (verified live). Exam data is fixed data: its defects are refused, never surprised by. | 4 |
| run | exam | the vision invariant: system as the ARGUMENT; run = grade ∘ compile. c7 deleted it (as_loss/CLI inlined grade∘compile) and 6 vision-shape tests broke: `run(benchmark, system)` IS the vision's headline shape — benchmark.run(system) is the api the optimizer consumes; inlining it makes every caller re-state the composition and the "system is the argument" law lives nowhere. | 1 |
| as_loss | export | the vision's headline: export the benchmark as a new loss function. c15 deletion-probed: removing it breaks the test collection itself (the import IS the pin) — the optimizer consumes loss(system) directly; it cannot be inlined away. | 1 |
| main | cli | s07 c3 proved CLI metal: a person runs `python -m nb` with no Python knowledge; owns the file layer since c4 (load deleted). c10 deleted the silent first-system default: the system taking the exam is NAMED, always — refusal beats surprise. c12 pinned the stdout ack line (pin broke on deletion): ONE human-readable line — which exam, which system, what score — the artifact file is for programs, the ack is for people. c18 (DONATED from the old spine, .staging/benchy/core.py OntologyPath — registry key == on-disk layout): the CLI refuses a benchmark whose declared path != requested path; a mismatched file is a broken exam install that would silently grade under the WRONG identity. | 2 |

survived = deletion attempts in push cycles (c15: every concept has now survived ≥1 probe — the engine is fully pinned).

## the acceptance bar (met)

/sentiment · 6 cases · good keyword stub → 1.0 · dumb constant stub → 0.5 ·
artifact = JSON per-case + aggregate · loss(dumb) > loss(good) · fully
offline · stdlib only · benchmark is pure data · one bundle directory.