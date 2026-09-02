# nb — the s09 (runner) design

Hypothesis under test: the only hard engineering in a benchmark engine is
TAKING the exam at scale. Task/scoring/data are trivial records; the runner
(fan-out, retries, resume, the artifact) is the metal.

## shape

One engine module + one CLI module, stdlib only:

- `nb/__init__.py` — `locate(path)`, `Exam` (`run(system)`, `as_loss(system)`,
  `fingerprint()`), plus private metal: `_score`, `_compile`, `_attempt`, `_write`.
- `nb/__main__.py` — `python -m nb <exam-or-path> <system> -o out.json` with
  exit code 0 iff zero errored cases. Runnable without pytest.
- `bench/hello/` — exam.json (path `/sentiment`, 6 cases, out vocabulary,
  optional scoring weights) + systems/good.json + systems/dumb.json. Pure data.
- Systems are DATA specs compiled by `_compile`: `always`, `keyword`, `flaky`
  (scripted failure + wrap of any other spec). The cloud exam-taker (steering
  addendum) is a future spec kind, not engine code.

## the runner contract (this angle's bar)

- fan-out over cases: ThreadPoolExecutor, default 8 workers.
- retries: each case gets `tries` attempts; exhausted attempts are LOUD
  (status error + message in the artifact), never silent zeros.
- kill-safety: every completed case triggers a full artifact rewrite
  (temp + atomic rename). O(n^2) writes are the price of durability; the
  artifact IS the resume contract (s04's verdict, our home turf).
- resume: re-run with the same `out` path keeps ok cases, re-attempts the
  rest. Content fingerprint (cases+scoring hash) refuses stale-evidence
  resume after an exam edit; system spec must match exactly.
- exit code: CLI exits 1 iff any case errored. The artifact is the truth,
  stdout is a convenience.

## concept table

| concept | pillar | why undeletable | survived |
|---|---|---|---|
| locate | DATA | ontology path `/sentiment` → exam dir; the vision's locateable-by-path | 0 |
| Exam | DATA (compressor) | carries the vision's method syntax: run(system)/as_loss(); the surface IS the spec | 0 |
| Exam.run | RUNNER | the exam-taking: fan-out + retries + resume + artifact; BARE_METAL c1: serial fails the 1000-case bar 16x over (25.9s vs 1.55s against a load-bearing bound — 10ms sleep/attempt, floor = n·tries·sleep, never undersleeps); threads (not asyncio, not raw threading) because systems are sync functions and ThreadPoolExecutor is the leanest stdlib fan-out | 1 |
| Exam.as_loss | SCORING | vision invariant: loss = 1 - score; ranks systems for optimizers | 0 |
| Exam.fingerprint | RUNNER | resume must not reuse stale evidence after an exam edit; content identity, not dir identity | 0 |
| _score | SCORING | shape-dispatch on want: scalar exact match / dict weighted; data-only scoring | 0 |
| _compile | SYSTEM | system specs are data; this is the compiler pillar's one function | 0 |
| _attempt | RUNNER | one case: invoke with retries, graded evidence record | 0 |
| _write | RUNNER | atomic artifact write; kill-safety is the resume contract | 0 |
| main | CLI | runnable without pytest; exit codes; resume via same -o path | 0 |

## cycle 1 verdict — concurrency is BARE_METAL (the angle's central question, answered)

Deleted ThreadPoolExecutor → serial against a bound made load-bearing FIRST
(the honest order: tighten the test until concurrency is the only way through,
then delete and watch it break). Serial floor = n · tries · sleep = 1000 · 2 ·
10ms = 20s of pure `time.sleep` — sleep never undersleeps, so the 15s bound
bites deterministically, not by CI luck. Result: serial 25.9s (FAILS),
threads 1.55s (16x). At real cloud exam-taker latency (100ms–2s/call) the gap
is 100x+. asyncio remains rejected: systems are plain sync
`invoke(input) -> prediction` — viral async would leak into the SYSTEM pillar.
Raw threading.Thread+Queue = strictly more LOC for the same guarantee.
ThreadPoolExecutor is the bare metal of fan-out here.

## open questions for push cycles

- does `fingerprint` survive as a public concept or fuse into run()? (c2)
- `tries`/`workers` defaults: named constants or noise? flag vs mechanism (c3)
- is `_write` temp+rename metal? the SIGKILL test is the judge (c4)