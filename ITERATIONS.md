# ITERATIONS.md — the golem log (one line per cycle)
# build: engine from zero under s07 (datacentric). 17 tests green, hello bar
# passing offline (good=1.0, dumb=0.5, as_loss ranks, /sentiment locatable).
1 | pushed=lenses-from-samples (inferred enum + hard-coded scoring, delete declared task/scoring) | broke=yes | verdict=BARE_METAL | loc=99 | concepts=9
2 | pushed=_check_exam (fuse validation into load, its only caller) | broke=no | verdict=HARD_PUSH | loc=98 | concepts=8
3 | pushed=main CLI (delete nb/__main__.py) | broke=yes | verdict=BARE_METAL | loc=97 | concepts=8
4 | pushed=context (sample key + invoke param + artifact field) | broke=no | verdict=HARD_PUSH | loc=97 | concepts=8
5 | pushed=as_loss (delete the loss-export) | broke=yes | verdict=BARE_METAL | loc=97 | concepts=8
6 | pushed=grade (fuse the scoring lens into run, its only caller) | broke=no | verdict=HARD_PUSH | loc=96 | concepts=7
7 | pushed=const kind (a constant IS keyword with any=[]) | broke=yes | verdict=HARD_PUSH | loc=94 | concepts=7
8 | pushed=_check required param (SAMPLE_REQUIRED==SAMPLE_ALLOWED) + escalate: drop kind from key set | broke=yes | verdict=HARD_PUSH | loc=92 | concepts=7
9 | pushed=locate double-read (probe vs entry) + run's duplicate policy gate | broke=yes | verdict=BARE_METAL | loc=89 | concepts=7
10 | pushed=sample id (write-only metadata: schema+engine+artifact+data, full delete) | broke=no | verdict=HARD_PUSH | loc=89 | concepts=7
11 | pushed=artifact echoes (path+system deleted — unread, callers hold their own args; escalated to per-case input) | broke=yes (input: a case must interpret alone) | verdict=BARE_METAL | loc=89 | concepts=7
12 | pushed=invoke's dict/kind gate (load already validates specs — duplicate?) | broke=yes (unknown-kind spec raised KeyError not ValueError) | verdict=BARE_METAL | loc=89 | concepts=7
13 | pushed=SCORE_KEYS indirection (one-entry set consulted once) | broke=no | verdict=HARD_PUSH | loc=87 | concepts=7
14 | pushed=EXAM_KEYS schema check (the five literal reads make the set redundant?) | broke=yes (unknown top key silently ignored) | verdict=BARE_METAL | loc=87 | concepts=7