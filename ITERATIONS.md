# ITERATIONS.md — the golem log (one line per cycle)
# build: engine from zero under s07 (datacentric). 17 tests green, hello bar
# passing offline (good=1.0, dumb=0.5, as_loss ranks, /sentiment locatable).
1 | pushed=lenses-from-samples (inferred enum + hard-coded scoring, delete declared task/scoring) | broke=yes | verdict=BARE_METAL | loc=99 | concepts=9
2 | pushed=_check_exam (fuse validation into load, its only caller) | broke=no | verdict=HARD_PUSH | loc=98 | concepts=8
3 | pushed=main CLI (delete nb/__main__.py) | broke=yes | verdict=BARE_METAL | loc=97 | concepts=8
4 | pushed=context (sample key + invoke param + artifact field) | broke=no | verdict=HARD_PUSH | loc=97 | concepts=8
5 | pushed=as_loss (delete the loss-export) | broke=yes | verdict=BARE_METAL | loc=97 | concepts=8