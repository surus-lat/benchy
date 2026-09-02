# s04 — onefile: the design

Hypothesis: the true engine fits in ONE file under 400 lines, stdlib only.
If it does not fit, the design is wrong, not the file.

## the shape

One file, `nb/benchy.py`, read top to bottom as the four pillars:

```
SYSTEM   invoke()          — the AI-API: one protocol f(in, ctx) -> out.
                            Systems are DATA (const | rule | py) or callables.
TASK     Task              — the program description: in -> out schema + ont path.
                            ok() gates enum outputs.
SCORING  grade()           — 'exact' | {'fields': weights}. importance hierarchy.
                            grade is the loss's atom.
DATA     Benchmark / Exam  — bench.json = task + cases + scoring + systems.
                            run(system) -> Exam (the graded artifact),
                            as_loss() -> the same exam as a float.
```

The invariants live in this shape with zero extra concepts:
- **resume**: `run(--out)` reads the artifact, keeps graded case rows, re-takes
  only the missing ones. Atomic write (tmp + os.replace) so a kill never
  corrupts the evidence. One loop, ~10 lines — not noise.
- **async fan-out**: `ThreadPoolExecutor` in `run()`. The pool IS the runner;
  there is no Runner concept.
- **artifacts**: the Exam serializes itself; the artifact IS the object.
  No writer/formatter layer.
- **as_loss()**: `lambda system: self.run(system).loss`. The benchmark is a
  loss function over systems; run() is its evidence trace.

Ontology: each bench declares `task.ont` = `/<task?>/<domain?>/<language?>`;
`Benchmark.load("/sentiment")` walks `bench/` for it. The registry is the
filesystem — no registry object.

## concept table

| concept | pillar | why undeletable | survived |
|---|---|---|---|
| invoke | SYSTEM | the single AI-API protocol; every system shape plugs in here. Model/node/workflow/agent all become f(in,ctx)->out. | 0 |
| Task.ok | TASK | enum gating: a prediction outside the output schema is not an answer. Vision: task defines the program's i/o. | 0 |
| Task.spec | TASK | carries in/out schema + ont path; the search-for-this-program description. | 0 |
| grade | SCORING | what good means; the atom of both score and loss. fields weights = importance hierarchy. | 0 |
| Exam | DATA | the graded evidence: per-case rows + aggregate. Its dict IS the artifact — no separate writer concept. | 0 |
| Exam.score/loss | SCORING | point estimate of the distribution + the loss view for optimizers. | 0 |
| Benchmark.run | DATA | the exam-taking loop; hosts resume + fan-out + artifact write — all one loop, no sub-concepts. | 0 |
| Benchmark.as_loss | SCORING | vision: benchmark = a new loss function for software-3.0. loss(dumb) > loss(good) ranks systems. | 0 |
| Benchmark.load | DATA | by dir / file / ontology path; the filesystem is the registry. | 0 |
| _write | DATA | atomic artifact persistence — resume's read side demands it; kill-safety. | 0 |
| main | CLI | run + loss; the UX. | 0 |
| _artifact | DATA | sorts rows + rebuilds Exam mid-run for the incremental write. | 0 |

## noise policy

Under the 400-line ceiling, every surviving concept must earn its lines.
The loudest things right now, pre-cycle-1: `Task.spec` wrapping (the Task
object may be just its dict), `_artifact` helper (fold into run?), the
`system`/`py` path-resolution branch in run(), the `name` derivation.