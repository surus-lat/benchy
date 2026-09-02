# The Ten Search Angles

Round 1 of the bare-metal search. Ten independent redesigns of benchy from
zero, one per worktree, all under the law of search/GOLEM.md, all running
the same hello benchmark (GOLEM.md) so the unify phase can compare them.

These briefs are STARTING HYPOTHESES, not cages. The metal wins over the
angle: if your search proves your angle wrong, follow the metal and say so
in LEARNINGS.md. A searcher that proves its own hypothesis wrong has still
succeeded — that is information.

All searchers: engine goes in `nb/`, tests in `nb_tests/`, hello benchmark
in `bench/hello/`. Read VISION.md and IDEAS.md before cycle 1.

---

## s01 — loss-first

Hypothesis: the benchmark IS a loss function over systems. `as_loss()` is
not a feature; it is the identity, and everything else is derived from it.
Start from `loss = benchmark.as_loss()`; make `run(system)` and the graded
artifact just the evidence trace of one loss evaluation.
Bare metal if: every other concept (runner, artifact, config, CLI) is a
projection of (Task, Data, Scoring, System) → float.
Falsified if: the exam/report side needs concepts the loss view cannot
generate.

## s02 — exam

Hypothesis: the metaphor IS the model. Exam, cases, taker, grade, retake
(resume), report card (artifact). Name every concept with exam words; if a
needed concept has no honest exam word, suspect it is noise.
Bare metal if: a non-engineer reads the code and understands it from the
metaphor alone.
Falsified if: exam words start stretching to cover framework noise
(a "Scheduler" is not a proctor) and the metaphor lies.

## s03 — compiler / AI-API

Hypothesis: IDEAS.md is right — the system side is a COMPILER: it makes any
learned program (model, node, workflow, agent, old-ML) take the exam. The
rest of benchy only ever sees `invoke(input) -> prediction`.
Define the AI-API protocol (one method); adapters are compiler backends.
Write two backends: the stub, and an openai-compatible HTTP backend (tested
against a mock server — stdlib urllib, no network, no keys).
Prove composition: a workflow = a system whose backend composes other
systems, with NO new core concept.
Bare metal if: workflows and agents are configuration, not code.
Falsified if: some real system cannot be expressed without leaking into core.

## s04 — onefile

Hypothesis: the true engine fits in ONE file under 400 lines, stdlib only.
If it does not fit, the design is wrong, not the file.
Write `nb/benchy.py` top to bottom. Every time it exceeds 400 lines, that is
a golem signal: some concept inside is noise.
Bare metal if: `python3 nb/benchy.py run bench/hello stub` works and the
file stays under 400 lines through every cycle.
Falsified if: a vision invariant (resume, async fan-out, artifacts) cannot
live in one file without becoming noise — then document exactly which and why.

## s05 — yaml / benchmark-as-directory

Hypothesis: a benchmark is a DIRECTORY: task.yaml + scoring.yaml + cases.* +
(optional) systems/*. The engine is a pure interpreter of directories.
Zero Python to define, run, and score a benchmark.
Define the directory format first (the schema), then the interpreter.
Bare metal if: defining the hello benchmark = writing two tiny files a
non-engineer could write, and the engine never requires user Python.
Falsified if: scoring of any real complexity forces Python into the
interface — then design the minimal escape hatch and record what it cost.

## s06 — protocol / contracts-first

Hypothesis: the product is the PROTOCOLS: Task, Scorer, System, Data plus
the records that flow between them. Implementations are trivial one-liners.
Write `nb/core.py` as pure types/protocols/docstrings with no behavior,
then make the hello benchmark work with the dumbest possible impls.
Bare metal if: swapping any implementation changes zero lines outside it,
and the public surface stays under ~10 concepts.
Falsified if: protocols multiply — more than ~10 public types is noise
wearing a contract's clothing.

## s07 — datacentric

Hypothesis: data is the exam; task and scoring are LENSES over samples, not
peer pillars. Start from the Sample record: (id, input, expected, context?).
Task = the input/output type declaration; scoring = the comparison policy.
Try deriving what you can from the data itself (inferring shape from cases)
— but kill any magic the moment it gets clever: explicit beats implicit.
Bare metal if: a benchmark is literally just data plus two tiny pure
functions.
Falsified if: inference becomes a cleverness engine (noise) or the data
cannot carry the task/scoring semantics honestly.

## s08 — cli

Hypothesis: the product is three commands:
  `benchy run <bench> <system> [--limit N]`
  `benchy new <name>` (scaffold a benchmark)
  `benchy report <run>`
Design the CLI first: write it as a fake that echoes what it WOULD do, then
make each word real. Every flag you cannot justify is a concept the design
failed to absorb.
Bare metal if: the entire UX fits in a tweet and still covers the vision
loop: create → run → grade → export as loss.
Falsified if: the CLI grows flags (each one is evidence the core shape is
wrong).

## s09 — runner

Hypothesis: the only hard engineering is TAKING the exam at scale: async
fan-out over cases, retries, resume, the artifact contract (the spirit of
run_outcome.json), exit codes. Task, scoring and data are trivial records.
Build the async runner first against a fake system; add task/scoring/data
only when the runner demands them.
Bare metal if: 1000 cases against a deliberately flaky stub system run
concurrently, resumable after a kill, zero lost work, one JSON artifact,
under ~300 LOC.
Falsified if: reliability genuinely requires frameworks — then the old
benchy was right about something, and you must say exactly what.

## s10 — salvage (the archaeologist)

Hypothesis: the old benchy knows things a from-zero search will miss.
Cycles 1–12: from zero, exactly like everyone else.
Final cycles: audit the OLD system, read-only, from the main checkout at
/Users/dobleefe/benchy/ — old engine `benchy/` and `src/`, the previous
session's spine candidate `.staging/benchy/`, the artifact contract in
`AGENTS.md`, and `docs/`. (Ignore the copies inside your own worktree;
read the main checkout.)
For each old concept ask: does it solve a failure my from-zero design
actually hits? Donate ONLY what survives a deletion attempt inside YOUR
design. Every donation gets a written reason in LEARNINGS.md.
Bare metal if: the final design stays from-zero clean plus a short list of
earned donations.
Falsified if: donations flood in — then from-zero was a fantasy and the
unify phase must know that.