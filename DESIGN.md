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
  `task.output` (declared choices). "takes a text, returns {pos,neg}" is data.
- **SCORING pillar** lives in the benchmark data: `scoring.rule` (match),
  `scoring.aggregate` (mean). Grading is a lookup, not a framework.
- **DATA pillar** is the cases list — the exam itself. n cases, each a
  (input, expected) pair. Everything is a distribution; the artifact reports
  the point estimate plus the per-case evidence.
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
| compile | system | the SYSTEM pillar: turns a spec into invoke(text)->pred; the only place the engine may grow (cloud kinds) | 0 |
| grade | scoring+seam | the seam where ANY callable takes the exam — real APIs, workflows, cached runs bypass compile. c3 fused it into run and the seam test broke: every system was forced through spec-compilation. The exam loop lives here, with raw invoke, not with specs. | 1 |
| run | exam | the vision invariant: system as the ARGUMENT; run = grade ∘ compile. c4 candidate — but it IS the vision shape; deleting it leaves as_loss calling grade(compile) directly and no single entry. | 0 |
| as_loss | export | the vision's headline: export the benchmark as a new loss function | 0 |
| main | cli | s07 c3 proved CLI metal: a person runs `python -m nb` with no Python knowledge; owns the file layer since c4 (load deleted) | 0 |

survived = deletion attempts in push cycles (this session: build only).

## the acceptance bar (met)

/sentiment · 6 cases · good keyword stub → 1.0 · dumb constant stub → 0.5 ·
artifact = JSON per-case + aggregate · loss(dumb) > loss(good) · fully
offline · stdlib only · benchmark is pure data · one bundle directory.