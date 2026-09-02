# s04 — onefile: the design

Hypothesis: the true engine fits in ONE file under 400 lines, stdlib only.
If it does not fit, the design is wrong, not the file.

## the shape

One file, `nb/benchy.py`, read top to bottom as the four pillars:

```
SYSTEM   invoke()          — the AI-API: one protocol f(in, ctx) -> out.
                            Systems are DATA (rule+default | py) or callables;
                            a default-only rule IS a constant system.
TASK     (no code)         — the task IS data: spec["task"] = in/out schema +
                            ont path. An out-of-enum prediction cannot match
                            any want, so grading already scores it 0 — the
                            enum is the declared output space for optimizers,
                            not a gate the grader needs. ok() deleted (c4).
SCORING  grade()           — 'exact' | {'fields': weights}. importance hierarchy.
                            grade is the loss's atom.
DATA     Benchmark/exam()  — bench.json = task + cases + scoring + systems.
                            run(system) -> the graded artifact dict,
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
| invoke | SYSTEM | the single AI-API protocol; every system shape plugs in here. Model/node/workflow/agent all become f(in,ctx)->out. const folded into default-only rule (c3). Rule+default data specs are the learned-program shapes it serves; py specs compile to a callable ONCE per exam in run() (c9 — per-case import reset module state, a real correctness bug). invoke() itself is now rule+default only. | 2 |
| grade | SCORING | what good means; the atom of both score and loss. fields weights = importance hierarchy. Enum gates are noise here: an out-of-enum pred cannot match any want (ok() deleted, c4). | 1 |
| Benchmark._exam | DATA | builds the graded artifact dict: per-case rows + aggregate + loss. The dict IS the exam — Exam class deleted as noise (c2), the module-level exam() fn fused into its only caller (c11, HARD_PUSH: nothing broke, concept count dropped 6→5). No separate writer concept. | 4 |
| Benchmark.run | DATA | the exam-taking loop; hosts resume + fan-out + artifact write — all one loop, no sub-concepts. py specs compile to a callable here, once per exam (moved from invoke in c9). The mid-run write is BARE METAL (c10): kill-safety is the resume contract — with only a final write, a kill leaves NO artifact and all graded work is lost (proved by test_kill_midrun_keeps_graded_work, written c10). | 3 |
| Benchmark.as_loss | SCORING | vision: benchmark = a new loss function for software-3.0. loss(dumb) > loss(good) ranks systems. | 0 |
| Benchmark.load | DATA | by dir / file / ontology path; the filesystem is the registry. Three-way resolution survived its c2 push — bench.json-in-dir is how non-engineers hand you a benchmark, ontology path is the vision's /<task?>/<domain?>/<language?> address, direct file is the degenerate case. | 1 |
| _write | DATA | atomic artifact persistence — resume's read side demands it; kill-safety. | 0 |
| main | CLI | run only (c12): the `loss` verb deleted — the run artifact already carries loss; as_loss() the METHOD is the vision law and stays, the second CLI verb was a duplicate interface. Flag parser deleted (c7): out is a positional, limit/workers are engine kwargs. | 2 |

## noise policy

Under the 400-line ceiling, every surviving concept must earn its lines.
Noise removed so far: Task class (c1), Exam class (c2), invoke's const
shape (c3), ok() enum gate (c4 — an out-of-enum prediction cannot match
any want, grading already scores it 0), the py path-resolution branch in
run() (c5 — py paths resolve from CWD like every data path), the `name`
derivation + system label in the artifact (c6 — the artifact path IS the
label), main()'s flag parser (c7 — out is a positional, limit/workers are
engine kwargs), Benchmark.spec/task/ont-property attributes (c8 — the
engine keeps only what it uses; task spec stays pure data in bench.json),
py per-case dynamic import (c9 — compile to a callable once per exam in
run(); per-case import reset module state: a REAL correctness bug).

Remaining loudest things, in attack order:
1. DONE c10 — the mid-run incremental write is BARE METAL (kill-safety is
   the resume contract; a final-write-only engine loses ALL graded work on
   a kill). The O(n²) rewrite is the price of durability. Kept.
2. DONE c11 — exam() module fn fused into Benchmark._exam (HARD_PUSH:
   nothing broke, concepts 6→5). The artifact-build lives with its only
   caller; `ont` no longer travels as an argument.
3. DONE c12 — CLI `loss` verb deleted (HARD_PUSH: nothing broke; the run
   artifact carries loss; as_loss() the method is vision law and stays).
   Remaining in this family: Benchmark.load's three-way resolution.
4. grade()'s "exact" string special-case vs fields-shape unification.