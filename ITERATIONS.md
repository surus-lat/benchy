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