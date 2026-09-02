# ITERATIONS — the golem log

# format: <n> | pushed=<target> | broke=yes|no | verdict=<...> | loc=<n> | concepts=<n>
# Setup (before cycle 1): engine drafted from zero in exam words, hello exam authored as
# 3 data files, 13 tests green, acceptance bar met (good=1.0, dumb=0.5, artifact written,
# loss ranks stubs). Golem baseline recorded after this line.
1 | pushed=Question+Page+combine+grade_keyword+GRADES | broke=yes | verdict=HARD_PUSH | loc=214 | concepts=15
2 | pushed=sit.grade_page(dup name) | broke=no | verdict=HARD_PUSH | loc=213 | concepts=13
3 | pushed=report() fused into sit() | broke=no | verdict=HARD_PUSH | loc=211 | concepts=13
4 | pushed=retake()+hall-dual-path | broke=yes | verdict=NOISE_REMOVED | loc=201 | concepts=12
5 | pushed=PageResult | broke=no | verdict=HARD_PUSH | loc=193 | concepts=11
6 | pushed=_scribble/_read_scribble→answers.json | broke=yes | verdict=NOISE_REMOVED | loc=188 | concepts=9