# LEARNINGS — s09 (runner angle): what the bare metal actually is

15 push cycles. Golem final: cycles=15, files=2, loc=121, deps=0,
concepts=7. Verdicts: BARE_METAL 5 (c1,c4,c5,c11,c14), HARD_PUSH 8,
NOISE_REMOVED 2. 16/16 tests green in ~8.6s. Engine: nb/ (2 files,
stdlib only), benchmark bench/sentiment/ = pure data.

The angle's hypothesis was CONFIRMED: in a benchmark engine the hard
engineering IS taking the exam at scale. Task/scoring/data collapsed to
trivial records; every cycle that broke something broke it on the runner
side. But the deepest metal found was not a runner mechanism — it was
that **every observable claim needs an adversarial judge or it isn't
enforced**. That lesson (c4, c13, c14, c15) outlived every mechanism.

## TASK (program: input → output)

- The task lives in the CASES: per-case input + want, plus an optional
  declared `out` vocabulary that the engine loudly enforces (a want
  outside `out` refuses at Exam construction — c6/c12 deleted the
  exam.json `task`/`in`/`path` fields: unread cargo; the enforced
  vocabulary + the wants ARE the schema).
- No type system, no prompt template, no config. A task you cannot
  express as (input, want) pairs is not this engine's task.
- The ontology is the DIRECTORY TREE: bench/sentiment/ literally IS
  /sentiment. c6 deleted the rglob walk + path field (two addresses for
  one thing, and a lie-tolerating one: any dir could claim any path).
  c12 deleted locate's `root` kwarg (unread cargo).

## SCORING

- as_loss is BARE_METAL (c5): loss = 1 − score is the vision's
  export-to-optimizer seam. Deleted → loss-ranking broke instantly.
  The benchmark IS a loss over systems; the artifact is the evidence
  trace of one evaluation.
- Weights must RANK (c10): a weighted branch that doesn't order
  right-on-critical above right-on-nice is decoration, not hierarchy.
  Enforced: same correct-field count, different fields → different loss.
- Shape-dispatch on want: scalar = exact match; dict = weighted mean with
  weights defaulting to 1 (unweighted dict-want = per-field mean — one
  shape, no special case).
- Score = sum/total, total = exam size (c8): the mid-run value is an
  honest lower bound (ungraded count 0), the final value the exact mean.
  The competing "total = remaining work" semantics died against its own
  judge: a partial artifact must interpret alone; progress is derivable
  (len(cases)/total), exam size is not.
- errors is NOT a stored field — a projection (sum status=="error")
  computed by its one real reader, the CLI (c8).

## DATA (the exam)

- exam.json is pure data: out vocabulary, optional scoring block, cases.
  No Python required to define a benchmark; the escape hatch is unused.
- Content identity > hashing > naming. c2 deleted the sha256 fingerprint:
  explicit data (scoring block echo + per-record input/want vs current
  cases) is strictly stronger — it refuses edits AND tolerates case
  ADDITIONS (the hash refused any change and nuked prior work).
  c7 deleted the exam-NAME echo: content identity subsumes it (resume
  across a renamed exam correctly keeps work).
- BUT (c11, the asymmetry): the artifact's `system` echo is BARE_METAL
  and NOT subsumable. Content identity subsumes exam-NAME identity but
  NOT system identity — the system is the loss's FREE VARIABLE; without
  the echo, resume silently mixes two exam-takers' evidence into one
  artifact. Name = where the exam lives (subsumable); system = WHAT took
  it (provenance, metal).

## SYSTEM (compiler / AI-API)

- A system is a DATA spec compiled by one function (_compile): always,
  keyword, flaky. The cloud exam-taker (steering addendum) is a future
  spec kind, not engine code.
- flaky is the load-bearing spec kind (c9): deleting it broke 4 runner
  tests instantly. A deterministic scriptable failure is the ONLY honest
  probe for retries/loud-errors/kill-resume. `fails` = plain data (first
  k attempts fail; always-fail = fails ≥ tries — no special encoding).
  The "FP" periodic-flake mini-language was noise: no test exercised it.
- flaky's optional `sleep` default survived its probe (c14): flaky specs
  are routinely written without sleep — the failure is the point,
  latency is optional.

## RUNNER (the angle's pillar — the metal)

- **Threads, not asyncio** (c1): 16x speedup (25.9s serial → 1.55s
  concurrent, 1000 cases), and asyncio is noise here — systems are
  plain sync `invoke(input) -> prediction`; viral async would leak into
  the SYSTEM pillar. ThreadPoolExecutor is the leanest stdlib fan-out.
  Raw threading.Thread+Queue = strictly more LOC for the same guarantee.
- **Two-sided concurrency bounds** (c14): an upper bound alone proves
  nothing — a mutant that skips the sleep passes "dt < 0.75×serial
  floor" trivially (instant "concurrency"). The lower bound (dt >
  serial_floor/workers × 0.5 — the ideal concurrent floor; sleep never
  undersleeps so nothing cheats from below) certifies the work happened.
  The sleep-skip mutant passed the old judge 5/5 and dies in 0.41s on
  the new one. c1's own evidence had this hole; c13's lesson applied
  backwards to it.
- **Resume refusal is BARE_METAL** (c2/c11): identity = system echo +
  scoring echo + per-record (input,want) vs current cases. Refuses
  changed cases, changed scoring, different system — tolerates added
  cases and renamed exams. The c8 judge caught a REAL bug: a no-op
  resume (nothing to do) never entered the loop and returned an
  artifact with NO score (as_loss KeyError). Now every resume returns
  the complete artifact with its aggregate.
- **Atomic + incremental write is BARE_METAL** (c4): the O(n²)
  whole-artifact rewrite after every completed case is the price of
  durability. plain write_text TRUNCATES then refills — an external
  reader (dashboard/agent polling mid-run) saw torn/empty JSON in
  128/2225 busy-polls, 4/4 runs. tmp+rename only exposes complete
  states. The SIGKILL judge was TOO WEAK (the kill almost never lands
  in the window); the READER-poll judge is the honest one. Atomicity
  serves THE READER, not just the kill.
- **Retries are load-bearing; flaky is their only honest probe** (c9):
  exhausted retries are LOUD (status error + message + tries count in
  the record — c15 pinned tries-as-evidence: exhausted ≠ zero tries),
  never silent zeros.
- **Defaults are values, not cargo** (c15): run()'s workers=8/tries=3.
  tries 3→1 caught by 3 tests; workers 8→1 SURVIVED the whole suite
  (every big test passed workers explicitly — a judge gap) until the
  default-workers judge ran the big exam with NO knobs. Deleting the
  defaults entirely broke 13 tests: `result = benchmark.run(system)`
  with no knobs IS the vision's surface. Named module constants, by
  contrast, were cargo (c3) — the VALUE is metal, the NAME was noise.
- **Exit code = operability, not quality** (c13): exit 1 iff any case
  errored. The score-gate mutation (exit 1 iff score<1) SURVIVED the
  whole suite — a real judge gap, closed by the dumb-stub discriminator:
  score 0.5, zero errors → exit 0 (a wrong-but-completed exam is a
  SUCCESSFUL run; grading is the artifact's job).

## the meta-lesson (for the unify phase)

Four times the suite failed to enforce a claim we were SURE it enforced:
SIGKILL-vs-reader (c4), score-gating (c13), the one-sided bound (c14),
the default workers value (c15). Each was found only by mutating the
engine and watching the mutant survive. The suite now exists because its
own holes were attacked. Whatever the unify phase keeps from s09, keep
this procedure with it: **sharpen the judge first, mutate the engine,
and only trust a claim its mutant cannot survive.**