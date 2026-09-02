# LEARNINGS — s03 compiler / AI-API

What the metal actually is, pillar by pillar, after 15 push cycles.

## TASK — a dict, uninterpreted

`task.json` = `{"in": "text", "out": "label"}`. That's it. The engine
NEVER reads it. The Task class died in cycle 9 (a behavior-free
two-attribute wrapper whose only customer was its own test). The task
is the exam's *subject declaration* — it tells the SYSTEM author what
the program is, not the engine. Bare metal: a two-key dict flowing
through untouched. The engine treats it like a poster on the exam-room
wall.

## SCORING — a dict + one free function

`scoring.json` = `{"mode": ..., "weights": ...}` plus the free
`score(spec, pred, expected) -> 0..1`. The Scoring class died in cycle
11 (it existed only to carry mode+weights). Three modes cover the
vision's "what good means": exact (1 point per match), partial (per-
field fraction), weighted (business hierarchy — critical fields
count). One function, ~25 loc, is the entire grading semantics of the
engine. The weighted-without-weights fallback (degrade to partial)
is the only cleverness that survived — because deleting it would lie
to the user about their own spec.

## DATA — a list of tuples

`cases.json` → `[(input, expected), ...]`. The Exam class died in
cycle 10, the Case class in cycle 8 (a class that was a tuple with
attribute access). The emptiness guard lives at load(), the one door
where data enters. Distribution → point estimate (the vision's "point
estimate of a distribution") is one line: the mean, inside run().
Bare metal: JSON → tuples → iterate. No schema engine, no validation
layer, no dataset abstraction.

## SYSTEM — the compiler, and the angle's vindication

`invoke(input) -> prediction`. One callable is the ENTIRE AI-API.
`compile_system(spec)` is the front door: a spec is DATA (a dict), the
backend is picked by `kind`, and out comes the callable. The core
(Task/Scoring/Exam/Benchmark) never sees past the callable.

The backends, all pure configuration:
- **stub**: `{rules: {pattern: label}, default}` — keyword table; and
  since cycle 13, ALSO the constant (`rules: {}`), because a constant
  is just a stub that never matches. The dumbest system and the
  offline system are one concept.
- **http**: openai-compatible over stdlib urllib — the real world.
- **chain**: steps are specs — a WORKFLOW is just another spec. No new
  core concept.
- **agent**: model + tools + budget in a spec; the model emits
  `["tool", name, arg]` or a final value; the controller loop is
  backend code. THE HEADLINE: a full tool-loop agent was absorbed as
  pure config in cycle 6 with ZERO core changes. The angle's
  falsification probe came back negative — the compiler view held.

**The angle SURVIVED.** Model, workflow, agent: all one thing —
specs compiled to one callable. Nothing leaked into the core.

## The classes→data dissolution pattern

This search's signature move, run five times: Task (9), Exam (10),
Scoring (11), Benchmark (15) — every class dissolved to the data it
carried plus a free function for the behavior. The pattern:

  class X(field) + method  →  the field (data) + free fn(data, arg)

What it teaches: the classes were never load-bearing. Each one
"broke" tests on deletion only because tests referenced the class's
NAME, not because behavior was lost. Fixing forward (rewriting tests
against the data + function shape) always succeeded. A class earns
its existence only when invariants need protecting (see BARE_METAL
below) — carrying data is not an invariant.

The endgame (cycle 15): the engine has ZERO classes. Pillars are
data; operations are free functions; nothing else exists.

## BARE_METAL badges earned

1. **as_loss** (cycle 15). Deleted → test_loss_ranks_stubs broke AND
   GOLEM law 6 explicitly names `loss = benchmark.as_loss()` as an
   unbreakable vision invariant. The benchmark-as-loss-function IS
   the product's reason to exist (the bridge to software 3.0 prompt
   optimizers). Restored in place. It is three lines: close over the
   bench, return `1 - run(bench, system)["score"]`. Three lines of
   metal that everything else is downstream of.
2. **score's mode dispatch** (implicitly, cycles 5–7): the partial
   fallback for a weightless weighted spec survived every push — it
   is the difference between honest grading and silent zeros.

## What Benchmark's death taught (cycle 15)

Deleting the Benchmark class changed NOTHING: `run(bench, system)`
and `as_loss(bench)` over a plain `(task, scoring, cases)` tuple are
indistinguishable in behavior and BETTER in honesty — the golem
suddenly sees `run` and `as_loss` as public concepts (a class hides
its methods from an LOC/concept audit). The exam was always just
three pieces of data; the class was a name for a tuple. The vision's
"Benchmark = Task + Data + Scoring" survives literally as tuple
structure.