# s10 iteration log — push cycles only (golem counts these lines).
# baseline (not a push cycle): nb engine (5 functions, stdlib only), hello as
# pure data, 15 invariant tests, golem PASS 3 files/112 loc/0 deps/7 concepts.
# archaeology (cycles ~13-15, read-only audit of the old benchy) is a LATER session.

1 | pushed=_score_one+contains-rule | broke=no | verdict=HARD_PUSH | loc=107 | concepts=6
2 | pushed=constant-kind->degenerate-keyword | broke=no | verdict=HARD_PUSH | loc=105 | concepts=6
3 | pushed=grade->run-fusion | broke=yes | verdict=BARE_METAL | loc=105 | concepts=6
4 | pushed=load(engine-purity) | broke=no | verdict=HARD_PUSH | loc=102 | concepts=5
# c5 corrected: 13 tests broke on the bundle move and were fixed forward —
# strict golem wording for broke-and-fixed-forward is NOISE_REMOVED, not HARD_PUSH.
5 | pushed=bundle-purity(bench+systems-in-one-dir) | broke=yes (13 tests, fixed forward) | verdict=NOISE_REMOVED | loc=102 | concepts=5
6 | pushed=__init__.re-export-shim | broke=yes (imports pinned the shim; fixed forward to nb.engine — CLI still works as namespace package) | verdict=NOISE_REMOVED | loc=89 | concepts=5
7 | pushed=run(inline-grade∘compile-into-as_loss+cli) | broke=yes (6 vision-shape tests: run IS benchmark.run(system)) | verdict=BARE_METAL | loc=89 | concepts=5
# c8: task-block probe broke only its own presence guard — the engine never
# read it: an unenforced claim. Per the 4-pillar steering the declaration is
# metal, so fixed FORWARD: grade now refuses exam keys outside task.output.choices.
# +2 engine loc under --allow-growth (refusion, not new concept). 17 tests.
8 | pushed=task-block(unenforced-claim) | broke=yes (presence guard only; engine never read it — lie fixed forward into a grade refusal) | verdict=NOISE_REMOVED | loc=92 | concepts=5
9 | pushed=case-index-field | broke=yes (artifact-shape test pinned the derivable field; fixed forward — position IS the id, s07 c10) | verdict=NOISE_REMOVED | loc=91 | concepts=5
10 | pushed=cli-silent-first-system-default | broke=yes (cli test pinned the default; fixed forward — refusal with exit 2, the system taking the exam is NAMED always; +1 net loc under --allow-growth) | verdict=NOISE_REMOVED | loc=92 | concepts=5
# c11: the pin (test_unknown_aggregate_rejected, committed at fa3f33c) broke the
# moment the aggregate refusal was deleted — the refusal is metal. Escalation:
# the two same-kind refusals (unknown rule / unknown aggregate) fused into ONE
# literal equality check — the scoring you declare must be EXACTLY the scoring
# implemented (also refuses extra declared-but-unread keys, the c8 lesson).
# loc 92->91, no growth. 18 tests.
11 | pushed=aggregate-refusal-probe + scoring-refusals-fusion | broke=yes (the c11 pin) | verdict=BARE_METAL | loc=91 | concepts=5
# c12: the CLI stdout ack line was claimed but untested (CLI tests checked exit
# code + artifact file only). Pinned first (score/system/exam all on the line),
# deletion probe broke the pin — BARE_METAL: a person with no Python knowledge
# needs one human-readable line; the artifact file is for programs, the ack is
# for people (s07 c3 proved CLI metal). 19 tests, no growth.
12 | pushed=cli-stdout-ack-line | broke=yes (the c12 pin) | verdict=BARE_METAL | loc=91 | concepts=5
# c13: compile's case folding (w.lower()/text.lower()) was silent engine magic —
# unclaimed, untested, unexercised by the exam (all 6 cases and all pos words are
# lowercase). Deleted to LITERAL matching: the SPEC carries case, explicit beats
# implicit (s07). Nothing broke; the magic is fully gone. loc 91->90. 19 tests.
13 | pushed=compile-case-folding(silent-magic) | broke=no (deleted entirely) | verdict=HARD_PUSH | loc=90 | concepts=5
# c14: GOLEM.md's bar says the dumb stub is "always pos" but only the 0.5 score
# was pinned. Behavior pinned first, then the deletion probe (hardcode "neg" as
# the no-match default, spec's `default` unread) broke ONLY the behavior pin —
# on a balanced exam always-neg also scores 0.5, so the score is blind to which
# side a constant picks. `default` is metal: the constant system is the
# degenerate keyword and the choice lives in the SPEC, as data. 20 tests.
14 | pushed=compile-default(hardcode-neg) | broke=yes (the behavior pin only — score-blind on balanced exams) | verdict=BARE_METAL | loc=90 | concepts=5