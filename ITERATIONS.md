# ITERATIONS — s03

# <n> | pushed=<target> | broke=yes|no | verdict=<NOISE_REMOVED|BARE_METAL|HARD_PUSH|TOO_SOFT> | loc=<n> | concepts=<n>
1 | pushed=Task.out_enum + load_systems | broke=no | verdict=HARD_PUSH | loc=262 | concepts=17
2 | pushed=run() + CLI dual role (systems verb) | broke=yes | verdict=NOISE_REMOVED | loc=258 | concepts=16
3 | pushed=module-level invoke() | broke=no | verdict=HARD_PUSH | loc=255 | concepts=15