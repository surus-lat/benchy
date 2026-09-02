# s08 — cli

Hypothesis (search/ANGLES.md): the product is three commands —
`run <bench> <system> [--limit N]`, `new <name>`, `report <run>`.  Design the
CLI first, make each word real.  Bare metal if the whole UX fits in a tweet
and covers create → run → grade → export as loss.  Falsified if the CLI grows
flags.

**Interim verdict (cycle 0): the CLI absorbs the whole vision loop with one
flag.**  The three commands landed as the spec, the engine shrank to fit
under them (exam.py carries only what the words demand).  Every word is real
and pinned by a test: `new` scaffolds a benchmark that is *immediately
runnable*, `run` grades and writes evidence, `report` re-reads it.  The one
flag, `--limit`, is the smoke valve against cloud spend (a money
justification, not convenience).  Export-as-loss lives at the API seam
(`exam.as_loss()`) because its consumer is a program (a prompt-optimizer),
not a human — the CLI prints the loss per run as the human-visible half.

## shape

    benchy new <name>        bench/<name>/{benchmark.json,systems.py} — a
                            runnable scaffold (zero cases -> loud refusal)
    benchy run <bench> <sys> [--limit N]
                            locate by ontology path, the system takes the
                            exam, graded artifact -> runs/<bench>-<sys>.json
    benchy report <run>      re-read a graded run: per-case pass/fail +
                            score + loss (evidence outlives the process)

    benchmark.json           {path, task, cases} — the whole exam, pure data;
                            unknown keys are loud (an exam is exactly this)
    systems.py               the exam-takers, NOT part of the benchmark:
                            a system is any invoked program; the cloud
                            taker (steering addendum) joins here as a spec
    Exam.run(system)         -> graded artifact {system, benchmark, task,
                            cases:[{input, expected, prediction, score}],
                            score}   (mean; interprets alone)
    Exam.as_loss()           -> (system) -> 1 - score, lower is better
    locate(bench_root, path) ontology path -> exam (the address IS the registry)

Errors are spoken words (`benchy: <reason>`), never tracebacks — a product
CLI speaks.  `python -m nb` until packaging earns the console script.

## concept table

| concept | pillar | why undeletable (so far) | survived |
|---|---|---|---|
| Exam | ALL | carries the vision invariant's own syntax — `benchmark.run(system)`, `benchmark.as_loss()` (GOLEM law 6) | 0 |
| locate | DATA | the ontology path `/<task?>/<domain?>/<language?>` is the vision's addressing scheme; the walk IS the registry | 0 |
| run | UX+SYSTEM+DATA | the take-the-exam command; binds taker from systems.py and writes evidence | 0 |
| new | DATA | CREATING benchmarks is benchy's focus (VISION p.2); the scaffold must be runnable, not a blank page | 0 |
| report | DATA | evidence outlives the process that made it; re-running a cloud system to see a grade costs money | 0 |
| main | UX | dispatch + words-not-tracebacks; a CLI that raises stack traces at users is not a product | 0 |

## flags

| flag | justification |
|---|---|
| `--limit N` | the smoke valve: grade on the first N cases before spending on a full cloud run — the old benchy's entire smoke workflow (AGENTS.md) reduced to one flag |

Every other shape the vision demands arrived as a *word*, not a flag.