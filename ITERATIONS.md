# s06 iterations — one line per push cycle (golem format)
# Baseline (cycle 0, not a push): core.py contracts (Task, Case, Scored,
# System, Scorer — zero behavior), exam.py loop (Exam/load/locate/exact_match),
# __main__ CLI, hello benchmark as pure data + two stub takers. 10 concepts,
# ~99 loc. pytest green, golem baseline recorded. Every cycle below pushes.1 | pushed=Scorer-Protocol-as-class (cycle-1 accidental discovery: declared protocol class, implemented as function, broke 7 tests) | broke=yes | verdict=NOISE_REMOVED | loc=93 | concepts=9
2 | pushed=Scorer-type-alias (named contract deleted; scoring = plain callable convention, seam survived duck-typed) | broke=yes (NameError in SCORINGS annotation, fixed forward) | verdict=NOISE_REMOVED | loc=89 | concepts=9
