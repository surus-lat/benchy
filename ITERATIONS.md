# ITERATIONS — s03

# <n> | pushed=<target> | broke=yes|no | verdict=<NOISE_REMOVED|BARE_METAL|HARD_PUSH|TOO_SOFT> | loc=<n> | concepts=<n>
1 | pushed=Task.out_enum + load_systems | broke=no | verdict=HARD_PUSH | loc=262 | concepts=17
2 | pushed=run() + CLI dual role (systems verb) | broke=yes | verdict=NOISE_REMOVED | loc=258 | concepts=16
3 | pushed=module-level invoke() | broke=no | verdict=HARD_PUSH | loc=255 | concepts=15
4 | pushed=load_system_specs + _read (fusion into compile_systems) | broke=no | verdict=HARD_PUSH | loc=254 | concepts=14
5 | pushed=_get (inlined into score) + dead failures counter | broke=no | verdict=HARD_PUSH | loc=249 | concepts=13
6 | pushed=agent probe (kind=agent backend, tools+loop as data) | broke=no | verdict=HARD_PUSH | loc=274 | concepts=14
# growth-for-probe, sanctioned by the angle brief: a full tool-loop agent was
# thrown at the compiler and absorbed ENTIRELY as data (one _BACKENDS entry,
# zero core changes) — the agent-as-core-concept is fully gone, nothing broke,
# the angle survived its falsification attempt.
7 | pushed=Scoring.as_loss (fused into Benchmark.as_loss) + Task.__repr__ | broke=no | verdict=HARD_PUSH | loc=269 | concepts=14
8 | pushed=Case class (plain tuples) + Exam.__len__ | broke=no | verdict=HARD_PUSH | loc=259 | concepts=13
9 | pushed=Task class (pillar survives as data on Benchmark.task) | broke=yes | verdict=NOISE_REMOVED | loc=250 | concepts=12
10 | pushed=Exam class (cases list is the exam; empty-guard moved to load) | broke=yes | verdict=NOISE_REMOVED | loc=238 | concepts=11
11 | pushed=Scoring class (pillar = scoring.json data + free score()) | broke=yes | verdict=NOISE_REMOVED | loc=236 | concepts=11
12 | pushed=compile_systems (inlined into main; tests carry own helper) | broke=yes | verdict=NOISE_REMOVED | loc=229 | concepts=10
13 | pushed=_backend_const (const == stub with empty rules; dumb-stub.json migrated) | broke=yes | verdict=NOISE_REMOVED | loc=224 | concepts=9
14 | pushed=load task=spec alias + main dual error branches (names-or-all folded into the for) | broke=no | verdict=HARD_PUSH | loc=218 | concepts=9
