# s08 iterations — one line per push cycle (golem format)
# Baseline (cycle 0, not a push): the build. CLI-first: three commands as the
# spec (run/new/report in cli.py), engine shrank under them (exam.py: Exam,
# locate), hello benchmark as pure data + systems.py stubs, 10 tests green,
# golem baseline recorded. The build counts as context; every cycle below pushes.
1 | pushed=artifact-"task"-field then escalated: Exam.task-param (task stays in benchmark.json as taker-facing data — the cloud compiler reads it; engine plumbing never did) | broke=no (both were lenses: derivable via the benchmark address) | verdict=HARD_PUSH | loc=116 | concepts=6
2 | pushed=flag-parser (5 lines -> 2, dict(zip(argv[3::2], argv[4::2]))) + unknown-flag speaks words not tracebacks (TypeError joins the speak-words tuple); false limit-claim cut from docstring | broke=no | verdict=HARD_PUSH | loc=115 | concepts=6
3 | pushed=SYSTEMS-scaffold-template (Todo stub) — scaffold is now pure data (law 5); empty-cases refusal moved to locate (load-time honesty); escalated: NotImplementedError left speak-tuple (dead), dir.name second address deleted (artifact = ontology path, runs/sentiment-good.json) | broke=yes (limit+report tests pinned dir.name) | verdict=NOISE_REMOVED | loc=104 | concepts=6