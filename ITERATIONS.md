# ITERATIONS — s09 (runner)

# build (cycle 1 context, not a push cycle): runner-first engine.
# nb/__init__.py (locate, Exam.run/as_loss/fingerprint, _score/_compile/_attempt/_write)
# + nb/__main__.py CLI + bench/hello exam+systems data + 13 tests, all green.
# hello bar: good=1.0 dumb=0.5, loss(dumb)>loss(good), /sentiment located, artifact
# interprets alone; runner bar: 1000 flaky cases concurrent, SIGKILL mid-run -> resume
# with zero lost work. ThreadPoolExecutor (not asyncio) won the build: systems are
# plain sync functions, viral async would leak into the SYSTEM pillar.