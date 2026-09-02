# s04 golem log — prose starts with #, cycles are bare lines
1 | pushed=Task class (fused to dict+ok()) | broke=yes | verdict=NOISE_REMOVED | loc=142 | concepts=7
2 | pushed=Exam class (fused: artifact dict IS the exam, exam() fn builds it) | broke=yes | verdict=NOISE_REMOVED | loc=138 | concepts=7
3 | pushed=invoke const shape (folded into default-only rule) | broke=yes | verdict=NOISE_REMOVED | loc=136 | concepts=7
4 | pushed=ok() gate (deleted: out-of-enum pred cannot match any want, grading already scores it 0) | broke=yes | verdict=NOISE_REMOVED | loc=132 | concepts=6
5 | pushed=py path-resolution branch in run() (deleted: py paths resolve from CWD like every data path) | broke=no | verdict=HARD_PUSH | loc=129 | concepts=6
6 | pushed=name derivation + system label in artifact (deleted: artifact path IS the label; nothing reads artifact["system"]) | broke=no | verdict=HARD_PUSH | loc=122 | concepts=6
7 | pushed=main() flag parser (deleted: out is a positional arg; limit/workers are engine kwargs, not a second CLI interface) | broke=yes | verdict=NOISE_REMOVED | loc=111 | concepts=6
8 | pushed=Benchmark.spec/task attributes + ont property (deleted: engine keeps only what it uses; task spec stays pure data in bench.json) | broke=no | verdict=HARD_PUSH | loc=110 | concepts=6
9 | pushed=py per-case dynamic import (moved to run(): compile py spec to callable ONCE per exam; invoke() now rule+default only) | broke=yes | verdict=NOISE_REMOVED | loc=108 | concepts=6
