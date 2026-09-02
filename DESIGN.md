# DESIGN — s03 compiler / AI-API

The system side is a COMPILER. The rest of benchy only ever sees
`invoke(input) -> prediction` — one method, the whole AI-API. Model,
node, workflow, agent, old-ML: all are BACKENDS that compile a learned
program into that one callable. Benchy grades what the callable
returns. Nothing else exists.

## the shape

```
TASK      Task(in, out)                     the program description
SCORING   Scoring(mode, weights).score()    what good means; also the loss
DATA      Exam([Case(input, expected)])     the exam; JSON on disk
SYSTEM    compile_system(spec) -> callable  the compiler front door
BENCH     Benchmark(t, s, e).run(system)    the system is the ARGUMENT
          Benchmark.as_loss() -> (System)->float   the exam AS a loss

a benchmark on disk = a directory:
    task.json + scoring.json + cases.json + systems/*.json
locatable by ontology path:  /<task?>/<domain?>/<language?>
```

## concept table

| concept | pillar | why undeletable | survived |
|---|---|---|---|
| Task | TASK | the program description (in→out types) is the thing searched for; without it the exam has no subject | 0 |
| Scoring | SCORING | grading is what makes a benchmark a loss function; deleting it leaves only raw predictions | 0 |
| Exam | DATA | the exam is the data; distribution → point estimate happens here | 0 |
| Case | DATA | one (input, expected); the atom of evidence | 0 |
| Scoring.score | SCORING | per-case 0..1; the only place grades happen | 0 |
| Scoring.as_loss | SCORING | the vision's headline: benchmark as new loss for optimizers | 0 |
| Benchmark | TASK+DATA+SCORING | the exam as one value; the system is its argument, not a field | 0 |
| Benchmark.run | BENCH | takes the exam: invoke per case, grade each, aggregate | 0 |
| Benchmark.as_loss | SCORING/BENCH | (System)->float; reusing one exam across many systems | 0 |
| compile_system | SYSTEM | the compiler front door: spec (data) → callable | 1 |
| invoke | SYSTEM | THE protocol: the callable itself — `system(input) -> pred`. Not a function; the shape of every backend's return value | 1 |
| _backend_stub | SYSTEM | keyword-table backend; makes the exam offline-runnable | 0 |
| _backend_const | SYSTEM | dumbest possible system; proves scoring discriminates (0.5) | 0 |
| _backend_http | SYSTEM | openai-compatible backend over stdlib urllib; the real world | 0 |
| _backend_chain | SYSTEM | workflow = system whose backend composes systems; no new concept | 0 |
| load | BENCH | benchmark as data on disk, locatable by ontology path | 2 |
| compile_systems | BENCH | systems/*.json -> {name: system}; the only systems door (raw-spec loading fused into compile) | 1 |
| main | BENCH | CLI: run a benchmark dir against its systems | 3 |

Removed in cycle 5: `_get` (one-line dict access inlined into `score`;
"unweighted weighted" now honestly reads as the partial fallback), the
dead `failures` counter in main() (never incremented — leftovers of a
design where a bad score was an error; a graded exam is data, not an
error).

Removed in cycle 4: `load_system_specs` (uncompiled raw-spec loader
fused into `compile_systems` — after the `systems` verb died nothing
consumed raw specs; compile-at-load is the compiler angle's law:
spec→callable happens at one door), `_read` (one-line JSON wrapper
inlined — a function that adds a name without adding a concept).

Removed in cycle 2: `run` (load.py convenience wrapper — nothing in the
engine called it; only an unused import kept it alive), the CLI `systems`
verb (display sugar — `ls` lists specs; the CLI's one job is to run the
exam), and the `run` verb (redundant once `systems` is gone: the command
is just `nb.load <bench> [system...]`).

Removed in cycle 1: `Task.out_enum/pred_enum` (premature enum magic; a
Task is just in→out type names — validation belongs to scoring, and
scoring's exact mode never needed it), `load_systems` (the compiled
loader — nothing used it; the CLI compiles specs at run time).

## falsification watch (angle brief)

Prove composition with NO new core concept: `_backend_chain` is just
another backend — a workflow is a system spec whose steps are system
specs. Agents (loop, tools) are the next probe: if they demand core
changes, the angle leaks, and I must say so in LEARNINGS.md.

Not-yet-proven suspects (will be pushed): the CLI's dual role (run vs
systems listing) — maybe noise. `load_systems` (compiled) vs
`load_system_specs` (raw) — maybe one is redundant. `Task.out_enum`
complexity hints the Task constructor may be over-built.