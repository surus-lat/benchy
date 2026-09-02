# s06 iterations — one line per push cycle (golem format)
# Baseline (cycle 0, not a push): core.py contracts (Task, Case, Scored,
# System, Scorer — zero behavior), exam.py loop (Exam/load/locate/exact_match),
# __main__ CLI, hello benchmark as pure data + two stub takers. 10 concepts,
# ~95 loc. pytest green, golem baseline recorded. Every cycle below pushes.
1 | pushed=Scorer-Protocol-class (accidental discovery: declared protocol class, implemented as function, broke 7 tests) | broke=yes | verdict=NOISE_REMOVED | loc=93 | concepts=9
2 | pushed=Scorer-type-alias (named contract deleted; scoring = plain callable convention, seam survived duck-typed) | broke=yes (NameError in SCORINGS annotation, fixed forward) | verdict=NOISE_REMOVED | loc=89 | concepts=9
3 | pushed=Scored-TypedDict then escalated: conforms-artifact-field + Task-type + Exam.task-param (first push TOO_SOFT, escalation deleted 3 concepts) | broke=yes (tests guarded removed surface; fixed forward — task/conforms were annotation-cargo) | verdict=HARD_PUSH | loc=77 | concepts=7