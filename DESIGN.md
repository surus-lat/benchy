# DESIGN — s03 compiler / AI-API

The system side is a COMPILER. The rest of benchy only ever sees
`invoke(input) -> prediction` — one method, the whole AI-API. Model,
node, workflow, agent, old-ML: all are BACKENDS that compile a learned
program into that one callable. Benchy grades what the callable
returns. Nothing else exists.

## the shape

```
TASK      {"in": ..., "out": ...}             task.json — the program description
SCORING   Scoring(mode, weights).score()    what good means (per-case 0..1)
DATA      Exam([(input, expected), ...])       the exam; JSON on disk
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
| task (data) | TASK | the task.json dict IS the program description (in→out); the engine never interprets it — the SYSTEM compiles against it; without it the exam has no subject. Its class wrapper died; the data survived | 1 |
| Scoring | SCORING | grading is what makes a benchmark a loss function; deleting it leaves only raw predictions | 0 |
| Exam | DATA | the exam is the data; distribution → point estimate happens here | 0 |
| Exam.__iter__ | DATA | run() iterates the exam — the exam's ONLY interface; a Case class was one attribute-access away from a tuple | 0 |
| Scoring.score | SCORING | per-case 0..1; the only place grades happen | 0 |
| Benchmark | TASK+DATA+SCORING | the exam as one value; the system is its argument, not a field | 0 |
| Benchmark.run | BENCH | takes the exam: invoke per case, grade each, aggregate | 0 |
| Benchmark.as_loss | SCORING/BENCH | (System)->float = 1 - run score; the exam AS a loss; the only loss (per-case aggregation lives in run's mean — a second as_loss was one-mean-away) | 0 |
| compile_system | SYSTEM | the compiler front door: spec (data) → callable | 1 |
| invoke | SYSTEM | THE protocol: the callable itself — `system(input) -> pred`. Not a function; the shape of every backend's return value | 1 |
| _backend_stub | SYSTEM | keyword-table backend; makes the exam offline-runnable | 0 |
| _backend_const | SYSTEM | dumbest possible system; proves scoring discriminates (0.5) | 0 |
| _backend_http | SYSTEM | openai-compatible backend over stdlib urllib; the real world | 0 |
| _backend_chain | SYSTEM | workflow = system whose backend composes systems; no new concept | 0 |
| _backend_agent | SYSTEM | agent = model + tools + budget in a spec; the tool LOOP is compiler code — the falsification probe that proved composition needs no core concept | 0 |
| load | BENCH | benchmark as data on disk, locatable by ontology path | 2 |
| compile_systems | BENCH | systems/*.json -> {name: system}; the only systems door (raw-spec loading fused into compile) | 1 |
| main | BENCH | CLI: run a benchmark dir against its systems | 3 |

## deletions

Removed in cycle 9: `Task` (the class — a behavior-free two-attribute
wrapper around the task.json dict; the TASK pillar survives as pure
data on Benchmark.task. Its only customer was its own test).

Removed in cycle 8: `Case` (a class that was a tuple with attribute
access — `(input, expected)` is the whole atom of evidence; dict→tuple
normalization moved to load(), where the JSON is read), `Exam.__len__`
(dead interface — run() iterates, tests count the artifact, nothing
ever asked the exam for its length).

Removed in cycle 7: `Scoring.as_loss` (fused into `Benchmark.as_loss` —
it was `1 - mean(scored_cases)`, and run() already computes exactly that
mean as `score`; two as_losses were one-mean-apart), `Task.__repr__`
(pure display sugar; nothing in engine or tests ever printed a Task).

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

PROBE RESULT (cycle 6): the agent SURVIVED as pure configuration.
`_backend_agent` (~25 loc) names a model (system spec), tools (system
specs), and a budget; the model emits `["tool", name, arg]` or a final
value; the controller loop is backend code. Task/Scoring/Exam/Benchmark
never learned anything. Zero core changes — the angle held: agents are
compiler backends, not engine concepts.

Not-yet-proven suspects (will be pushed): `Exam` (after Case died it is a
non-empty list + __iter__ — the emptiness check may belong to load),
`Scoring` the class (carries mode+weights; a free score() function over
the tuple may be the same concept with less machinery), the CLI's
`compile_systems` (single-customer helper), `main` argv plumbing.