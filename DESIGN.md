# DESIGN — s03 compiler / AI-API

The system side is a COMPILER. The rest of benchy only ever sees
`invoke(input) -> prediction` — one method, the whole AI-API. Model,
node, workflow, agent, old-ML: all are BACKENDS that compile a learned
program into that one callable. Benchy grades what the callable
returns. Nothing else exists.

## the shape

```
TASK      {"in": ..., "out": ...}             task.json — the program description
SCORING   score(spec, pred, expected) -> 0..1   spec = scoring.json dict (data)
DATA      [(input, expected), ...]          cases.json — the exam; JSON on disk
SYSTEM    compile_system(spec) -> callable  the compiler front door
BENCH     run(bench, system) -> artifact    bench = (task, scoring, cases) tuple;
          as_loss(bench) -> (System)->float  the system is always the ARGUMENT

a benchmark on disk = a directory:
    task.json + scoring.json + cases.json + systems/*.json
locatable by ontology path:  /<task?>/<domain?>/<language?>
```

## concept table

| concept | pillar | why undeletable | survived |
|---|---|---|---|
| task (data) | TASK | the task.json dict IS the program description (in→out); the engine never interprets it — the SYSTEM compiles against it; without it the exam has no subject. Its class wrapper died; the data survived | 1 |
| scoring (data) + score() | SCORING | the scoring.json dict IS the declaration (mode, weights); the free score(spec, pred, expected) -> 0..1 is the only place grades happen; grading is what makes a benchmark a loss function. The Scoring class that carried the dict died in cycle 11 | 1 |
| exam (data) | DATA | the cases list [(input, expected), ...] IS the exam — distribution → point estimate happens in run's mean; the Exam class that wrapped it (empty-check + __iter__) died in cycle 10 | 1 |
| run | TASK+DATA+SCORING | takes the exam: invoke per case, grade each, aggregate to the mean (distribution → point estimate). Was Benchmark.run; the class wrapper died in cycle 15 — free function over the (task, scoring, cases) tuple | 0 |
| as_loss | SCORING/BENCH | (System)->float = 1 - run score; the exam AS a loss. **BARE_METAL badge (cycle 15)**: deleting it broke the loss-ranks-stubs acceptance bar AND GOLEM law 6 names `loss = benchmark.as_loss()` as an unbreakable vision invariant — the loss view IS the product. Restored immediately | 1 |
| compile_system | SYSTEM | the compiler front door: spec (data) → callable | 1 |
| invoke | SYSTEM | THE protocol: the callable itself — `system(input) -> pred`. Not a function; the shape of every backend's return value | 1 |
| _backend_stub | SYSTEM | keyword-table backend; makes the exam offline-runnable. Absorbed `const` in cycle 13 (rules:{} + default == a constant), so it is also the dumbest possible system | 1 |
| _backend_http | SYSTEM | openai-compatible backend over stdlib urllib; the real world | 0 |
| _backend_chain | SYSTEM | workflow = system whose backend composes systems; no new concept | 0 |
| _backend_agent | SYSTEM | agent = model + tools + budget in a spec; the tool LOOP is compiler code — the falsification probe that proved composition needs no core concept | 0 |
| load | BENCH | benchmark as data on disk, locatable by ontology path | 3 |
| main | BENCH | CLI: run a benchmark dir against its systems (the systems-dir compile is inlined here — the CLI is the only engine customer of systems/*.json). Cycle 14 killed its dual error branches: empty systems dir is a quiet zero-exam, the only error is an unknown name | 4 |

## deletions

Removed in cycle 15: `as_loss` was ATTEMPTED first — it broke
test_loss_ranks_stubs (the hello acceptance bar: loss(dumb) > loss(good))
and GOLEM law 6 names it as an unbreakable vision invariant → restored =
**BARE_METAL badge**. Escalation then deleted `Benchmark` the class
entirely: the exam became a plain (task, scoring, cases) tuple and its
two methods became the free functions `run(bench, system)` and
`as_loss(bench)`. Zero behavior change, 23/23 still green, loc 218→207.
The engine's LAST class is gone — the classes→data dissolution is
total: the pillars survive as pure data, the operations as free
functions, and nothing else exists. (The one visible-concept growth
was sanctioned: methods that a class hides from the golem became
honest public surface.)

Removed in cycle 14: `load`'s `task = spec` alias (the TASK pillar is
the dict passed straight to Benchmark — an alias line is not a
concept), `main`'s empty-`names` branch and empty-`systems` branch
(`names or sorted(systems)` folds the first into the for; a benchmark
with zero systems is a quiet zero-exam, not an error — the only CLI
error left is an unknown name).

Removed in cycle 13: `_backend_const` (a constant IS a stub with empty
rules — `{"kind":"stub","rules":{},"default":X}`; four tests carried
const-kind specs and broke; dumb-stub.json migrated). The four backends
are now three: stub (absorbs const), http, chain, agent.

Removed in cycle 12: `compile_systems` (single-customer helper — main
was the engine's only caller; the dir-scan compile is now one dict
comprehension inline in main. Tests carry their own _systems()
convenience; test noise is not engine noise).

Removed in cycle 11: `Scoring` (the class — it existed only to carry
mode+weights, a two-field wrapper around the scoring.json dict; the
pillar survives as data on Benchmark.scoring + the free score()
function; methods were never separate golem concepts).

Removed in cycle 10: `Exam` (the class — after Case died it was a
non-empty check + __iter__; the emptiness guard moved to load(), the
only door where data enters; Benchmark.exam is the plain cases list,
which iterates identically), `nb/data.py` (the file existed to hold
the class).

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