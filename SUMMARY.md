# SUMMARY — s09 (runner angle), 15 push cycles, golem-clean finish

## final shapes

- `nb/__init__.py` (155 lines, loc 102): `locate(path)` — the tree IS the
  ontology (bench/sentiment/ = /sentiment, no walk, no path field);
  `Exam(dir)`: `run(system, out=None, workers=8, tries=3)` → artifact
  {system, scoring, total, cases[], score}; `as_loss(system, **kw)` =
  1 − run(**kw)[score]; `_score` (shape-dispatch, weighted RANKS),
  `_compile` (always|keyword|flaky fails-count — specs are data),
  `_attempt`, `_write` (tmp+rename atomic).
- `nb/__main__.py` (33 lines, loc 19): CLI, `-o` required, errors =
  projection computed here (its one reader), exit 1 iff errors.
- `bench/sentiment/`: exam.json (6 cases, out vocab, scoring block) +
  systems/good.json + dumb.json. Pure data, git-tracked.
- Runner metal: ThreadPoolExecutor fan-out, retries loud, resume by
  content identity (system+scoring echo + per-record input/want),
  atomic incremental rewrite after every case.

## metrics (golem report, verbatim)

{
  "cycles": 15,
  "verdicts": {
    "BARE_METAL": 5,
    "HARD_PUSH": 8,
    "NOISE_REMOVED": 2
  },
  "files": 2,
  "loc": 121,
  "deps": 0,
  "concepts": 7
}

16/16 tests in ~8.6s (16 after c14/c15 judges; was 15 on arrival).
loc 144 → 121 across the search; engine stdlib-only; benchmark = data.

## best discovery

**Content identity subsumes exam-NAME identity but NOT system identity**
(c7 vs c11). Resume identity = system echo + scoring echo + per-record
(input,want) vs current cases. The name is WHERE the exam lives —
subsumable. The system is the loss's FREE VARIABLE — the echo is the
provenance record; without it resume silently mixes two exam-takers'
evidence. One asymmetry found by deleting each identity separately.

## most expensive mistake

Trusting one-sided evidence. The c1 concurrency bound (dt < serial floor)
passed a sleep-skipping mutant 5/5; the c13 score-gate mutant survived
the WHOLE suite; c15's workers=1 default mutant survived too. Four judge
gaps (c4 SIGKILL-vs-reader, c13, c14, c15) — each found only by mutating
the engine and watching the mutant survive. Sharpening judges first cost
~4 cycles; not sharpening would have shipped four unenforced claims.

## to the other nine searchers

1. Make the probe deterministic BEFORE deleting: sleep never undersleeps
   → two-sided time bounds (upper proves fan-out, lower certifies the
   work). A bound a mutant can beat proves nothing.
2. Atomicity serves THE READER, not the kill: our SIGKILL test passed a
   torn-write mutant; the external busy-poll reader caught it 128/2225.
3. Mutate your own engine before trusting the suite. Every "surely this
   is enforced" claim we tested that way had a hole the first time.
4. Exit code = operability, not quality (errors-gate, never score-gate).
5. defaults' VALUES are metal (run(system) is the vision's surface);
   default NAMES/constants are cargo. Probe both separately.
6. flaky (deterministic fails-count) is the retry path's only honest
   probe — without a failing system, retries are claimed-but-untested.