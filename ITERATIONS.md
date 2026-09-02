# s06 iterations — one line per push cycle (golem format)
# Baseline (cycle 0, not a push): core.py contracts (Task, Case, Scored,
# System, Scorer — zero behavior), exam.py loop (Exam/load/locate/exact_match),
# __main__ CLI, hello benchmark as pure data + two stub takers. 10 concepts,
# ~95 loc. pytest green, golem baseline recorded. Every cycle below pushes.
1 | pushed=Scorer-Protocol-class (accidental discovery: declared protocol class, implemented as function, broke 7 tests) | broke=yes | verdict=NOISE_REMOVED | loc=93 | concepts=9
2 | pushed=Scorer-type-alias (named contract deleted; scoring = plain callable convention, seam survived duck-typed) | broke=yes (NameError in SCORINGS annotation, fixed forward) | verdict=NOISE_REMOVED | loc=89 | concepts=9
3 | pushed=Scored-TypedDict then escalated: conforms-artifact-field + Task-type + Exam.task-param (first push TOO_SOFT, escalation deleted 3 concepts) | broke=yes (tests guarded removed surface; fixed forward — task/conforms were annotation-cargo) | verdict=HARD_PUSH | loc=77 | concepts=7
4 | pushed=locate (delete ontology search) | broke=yes (ontology test: path /sentiment -> exam is a vision invariant, not directory convention) | verdict=BARE_METAL | loc=77 | concepts=7
5 | pushed=System-Protocol then escalated: Case-TypedDict + all typed annotations (core.py is now a conventions docstring, zero classes) | broke=no (deletion consumed its own only test; structural conformance was annotation-cargo) | verdict=HARD_PUSH | loc=70 | concepts=5
6 | pushed=SCORINGS registry (scoring kind hardcoded in load; a registry of one entry was a fake choice, custom scoring rides the injected-scorer seam) | broke=no | verdict=HARD_PUSH | loc=69 | concepts=5
7 | pushed=scoring-kind-in-data (benchmark.json "scoring" block: dead data after registry deletion — dangling kind-string pointing at nothing; same data-noise disease s05 found) | broke=no | verdict=HARD_PUSH | loc=69 | concepts=5
8 | pushed=as_loss (delete the loss export) | broke=yes (test_loss_ranks_dumb_above_good: AttributeError — the acceptance bar itself demands as_loss() ranking the stubs) | verdict=BARE_METAL | loc=69 | concepts=5
9 | pushed=load (fuse into locate: the ontology path becomes the one address; locate the only constructor; Exam carries its source dir for the CLI) | broke=no (tests updated to the located seam — locate was already the entry point in every test path) | verdict=HARD_PUSH | loc=69 | concepts=4