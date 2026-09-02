# s05 iterations — golem log

1 | pushed=run.py+main (CLI as separate file/concept) | broke=yes | verdict=NOISE_REMOVED | loc=72 | concepts=9
2 | pushed=grade+_read_json+_read_jsonl (3 concepts fused/indlined into run/load) | broke=no | verdict=HARD_PUSH | loc=68 | concepts=6
3 | pushed=load_system (fused into load, optional system param) | broke=yes | verdict=NOISE_REMOVED | loc=65 | concepts=5
4 | pushed=task.json in/out keys (unread schema noise in data format; deletion law extends into the DATA) | broke=no | verdict=HARD_PUSH | loc=65 | concepts=5
5 | pushed=scoring.json aggregate key (unread; whole-dict loud check added) | broke=no | verdict=HARD_PUSH | loc=65 | concepts=5
6 | pushed=weighted scoring probe (fields+weights vocabulary; points key deleted as pinned constant) | broke=no | verdict=HARD_PUSH | loc=72 | concepts=6
7 | pushed=task.json itself (ontology locator; 11/12 tests broke — restored, fusion rejected) | broke=yes | verdict=BARE_METAL | loc=72 | concepts=6
8 | pushed=scoring.json itself (11/12 broke — C6 made it load-bearing again; weights are real interpreted data) | broke=yes | verdict=BARE_METAL | loc=72 | concepts=6