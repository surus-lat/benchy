# ITERATIONS

1 | pushed=SCORES/AGGS vocab breadth + evaluate/run fusion + mod-dance in system() | broke=yes | verdict=NOISE_REMOVED | loc=53 | concepts=3
2 | pushed=Bench class -> load() returns the loss itself | broke=yes | verdict=NOISE_REMOVED | loc=47 | concepts=3
3 | pushed=benchmark() fused into load() — the raw-spec second entry point | broke=no | verdict=HARD_PUSH | loc=46 | concepts=2
4 | pushed=system() short-name glob fallback — deleted, path is the only address | broke=no | verdict=HARD_PUSH | loc=45 | concepts=2
5 | pushed=SCORES/AGGS tables — deleted, inline honest scoring with loud vocab check | broke=no | verdict=HARD_PUSH | loc=44 | concepts=2
6 | pushed=receipt `i` key + spec.get(cases) default — both deleted, order is the index, missing cases must fail loud | broke=no | verdict=HARD_PUSH | loc=43 | concepts=2
7 | pushed=`task` key in bench.json — deleted, schema is visible in the cases (in values are the input type, want values are the output vocab); engine never read it | broke=no | verdict=HARD_PUSH | loc=43 | concepts=2
8 | pushed=`system()` concept — deleted, stdlib importlib is the loader, a system is just a callable; tests fixed forward | broke=yes | verdict=NOISE_REMOVED | loc=36 | concepts=1
9 | pushed=path key in loss.trace — evidence-only artifact | broke=yes | verdict=NOISE_REMOVED | loc=35 | concepts=1
10 | pushed=`want` key in trace cases — deleted, tests stayed green, but a failed case reads "wrong, about what?" without it; artifact must be self-contained for optimizers | broke=yes | verdict=BARE_METAL | loc=35 | concepts=1
11 | pushed=`scoring` key + engine check — deleted, stayed green (guard was unguarded); restored: scoring is a pillar (law #6), the check was live honesty code, vision promises multiple scorings; added the missing guard test | broke=no | verdict=BARE_METAL | loc=35 | concepts=1
12 | pushed=ROOT module anchor — inlined into load(); escalated: tried hoisting loss() out of the closure — broke 8 tests (spec needed at eval time, trace needs per-instance state); restored closure, added two-benchmarks-independent-receipts test | broke=yes | verdict=BARE_METAL | loc=34 | concepts=1
13 | pushed=loss.trace home — moved the receipt into the RETURN value (loss -> (float, trace)); broke 7 tests (pure-float contract, artifact, independence); no alternative home survives (kwarg leaks the interface, module-state clobbers); attribute restored | broke=yes | verdict=BARE_METAL | loc=34 | concepts=1