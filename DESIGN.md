# s08 — cli

Hypothesis (search/ANGLES.md): the product is three commands —
`run <bench> <system> [--limit N]`, `new <name>`, `report <run>`.  Design the
CLI first, make each word real.  Bare metal if the whole UX fits in a tweet
and covers create → run → grade → export as loss.  Falsified if the CLI grows
flags.

**Interim verdict (cycle 4): the CLI absorbed the whole vision loop with two
commands and one flag.**  The words landed as the spec, the engine shrank to
fit under them (exam.py carries only what the words demand).  `new` scaffolds
a benchmark as pure data that is *immediately honest* (zero cases -> loud
refusal at load), `run` grades and writes evidence.  The one flag, `--limit`,
is the smoke valve against cloud spend (a money justification, not
convenience).  Export-as-loss lives at the API seam (`exam.as_loss()`)
because its consumer is a program (a prompt-optimizer), not a human — the
CLI prints the loss per run as the human-visible half.  `report` was deleted
in cycle 4: the graded artifact IS the report — JSON that interprets alone —
and the vision loop (create → run → grade → export as loss) has no report
word in it.

## shape

```
benchy new <name>        bench/<name>/benchmark.json — a runnable scaffold
                            as pure data (zero cases -> loud refusal at load)
benchy run <bench> <sys> [--limit N]
                            locate by ontology path (the DIRECTORY is the
                            address), the system takes the exam, graded
                            artifact -> runs/<path>-<sys>.json
                            (the ontology path IS the artifact identity)

benchmark.json           {task, cases} — the whole exam, pure data; EXACTLY these
                            keys, enforced (cycle 9: the format was a docstring
                            claim until a test-first invariant failed against
                            the engine; one message names missing AND junk
                            keys).  The DIRECTORY IS the ontology path
                            (cycle 6 deleted the `path` data field — a second
                            address; the walk existed only to reconcile it).
                            task stays as taker-facing data: the cloud
                            compiler reads it to build prompts; the engine
                            passes it through untouched (cycle 1 deleted the
                            engine's own task plumbing — a lens); its
                            presence is enforced, its CONTENT is free (the
                            scaffold teaches a blank {} — any typed example
                            is a type-lie the engine does not check)
systems.py               the exam-takers, NOT part of the benchmark:
                            a system is any invoked program; the cloud
                            taker (steering addendum) joins here as a spec
Exam.run(system)         -> graded artifact {system, benchmark, cases:
                            [{input, expected, prediction, score}], total,
                            score} (mean; interprets alone — and names its
                            own scope: total says graded-of-total, so a smoke
                            run cannot masquerade as a full run, c10)
Exam.as_loss()           -> (system) -> 1 - score, lower is better
locate(bench_root, path) ontology path -> exam (the address IS the registry);
                            load-time honesty: unknown keys and zero cases
                            are refused before anything runs
```

Errors are spoken words (`benchy: <reason>`), never tracebacks — a product
CLI speaks.  `python -m nb` until packaging earns the console script.

## concept table

| concept | pillar | why undeletable (so far) | survived |
|---|---|---|---|
| Exam | ALL | carries the vision invariant's own syntax — `benchmark.run(system)`, `benchmark.as_loss()` (GOLEM law 6); cycle 5 dissolution broke both; cycle 7: grading is PURE (cases + scorer) — path/dir address bookkeeping died as lenses (identity composes at the write, in the CLI) | 2 |
| locate | DATA | the ontology path `/<task?>/<domain?>/<language?>` is the vision's addressing scheme — the DIRECTORY is the address; flat lookup, load-time honesty (unknown keys / zero cases refused before anything runs); the walk+`path`-field died in c6 (a registry re-deriving the address is noise) | 1 |
| run | UX+SYSTEM+DATA | the take-the-exam command; binds taker from systems.py and writes evidence — artifact identity (system/benchmark fields) composes HERE (c7 BARE_METAL: a JSON that leans on its filename does not interpret alone) | 1 |
| new | DATA | CREATING benchmarks is benchy's focus (VISION p.2); the scaffold is pure data — one file, honest refusal at load; c9: teaches a BLANK task {} (a typed example is a type-lie the engine does not check) and the ack line teaches the case shape in words | 0 |
| (main dissolved — cycle 8) | UX | dispatch is module code in __main__.py; if/elif over two verbs IS the table; tests drive the real process (subprocess), so the argv seam was dead weight | — |

## flags

| flag | justification |
|---|---|
| `--limit N` | the smoke valve: grade on the first N cases before spending on a full cloud run — the old benchy's entire smoke workflow (AGENTS.md) reduced to one flag; c10: the artifact carries `total`, so a smoke run can never masquerade as a full run (the artifact interprets alone, including its own scope) |

Every other shape the vision demands arrived as a *word*, not a flag.