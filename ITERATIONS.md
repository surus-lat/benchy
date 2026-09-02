# ITERATIONS

1 | pushed=SCORES/AGGS vocab breadth + evaluate/run fusion + mod-dance in system() | broke=yes | verdict=NOISE_REMOVED | loc=53 | concepts=3
2 | pushed=Bench class -> load() returns the loss itself | broke=yes | verdict=NOISE_REMOVED | loc=47 | concepts=3
3 | pushed=benchmark() fused into load() — the raw-spec second entry point | broke=no | verdict=HARD_PUSH | loc=46 | concepts=2
4 | pushed=system() short-name glob fallback — deleted, path is the only address | broke=no | verdict=HARD_PUSH | loc=45 | concepts=2
5 | pushed=SCORES/AGGS tables — deleted, inline honest scoring with loud vocab check | broke=no | verdict=HARD_PUSH | loc=44 | concepts=2