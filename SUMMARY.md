# s08 SUMMARY — cli angle

## metrics (golem report verbatim)

```json
{"cycles": 15, "verdicts": {"HARD_PUSH": 5, "NOISE_REMOVED": 7,
 "BARE_METAL": 3}, "files": 4, "loc": 89, "deps": 0, "concepts": 4}
```

## the two commands as landed (the whole UX)

    benchy new <name>       -> bench/<name>/benchmark.json
                               {task: {}, cases: []} — pure data, refuses overwrite
    benchy run <bench> <system> [--limit N]
                             -> runs/<path>-<sys>.json {system, benchmark,
                                total, cases:[{input, expected, prediction,
                                score}], score, loss}

Engine: nb/ (cli.py, exam.py, __init__.py, __main__.py), 89 LOC, 4
concepts (Exam, locate, new, run), 0 deps, 16 tests green.

## best discovery

**Law (a): every deletion that broke revealed a SECOND ADDRESS** —
the same fact stored twice is the noise's hiding place. Five
instances: dir.name (c3), `path` field (c6), filename-as-identity
(c7), loss formula at THREE addresses (c13), docstring duplication
(c14).
**Law (b): every probe that lied was a claim the engine never
enforced** — four instances, all test-first: scaffold shape (c9),
limit honesty (c10), per-case contract (c14 — died mid-exam, cloud
money), overwrite refusal (c14).

## most expensive mistake

`report` (cycles 0–4): a whole command for a job the graded
artifact already does; JSON that interprets alone IS the report.
~15 LOC + a UX word the vision loop doesn't contain. Deleted c4.

## what I would tell the other nine searchers

1. Design the CLI as the spec FIRST; the engine shrinks to fit.
2. When a deletion breaks, hunt the SECOND ADDRESS first.
3. Enforce every data claim at LOAD — never mid-exam (cloud money).
4. Scoring derives from the output schema; weights are LOUD data.
5. The artifact is all evidence: no write-only fields, identity
   and loss each composed once, stdout only echoes.
6. Delete scaffold typed examples (task-lie c9); survived=0
   concepts are unproven holes — deletion-test them (c15).
7. runpy beats importlib ceremony; main() is optional (c8).