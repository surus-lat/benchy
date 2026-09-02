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
| Exam.run | RUNNER | the exam-taking: fan-out + retries + resume + artifact | 0 |
| Exam.as_loss | SCORING | vision invariant: loss = 1 - score; ranks systems for optimizers | 0 |
| Exam.fingerprint | RUNNER | resume must not reuse stale evidence after an exam edit; content identity, not dir identity | 0 |
| _score | SCORING | shape-dispatch on want: scalar exact match / dict weighted; data-only scoring | 0 |
| _compile | SYSTEM | system specs are data; this is the compiler pillar's one function | 0 |
| _attempt | RUNNER | one case: invoke with retries, graded evidence record | 0 |
| _write | RUNNER | atomic artifact write; kill-safety is the resume contract | 0 |
| main | CLI | runnable without pytest; exit codes; resume via same -o path | 0 |

## open questions for push cycles

- is ThreadPoolExecutor metal or noise at this scale? (threads won the build;
  the cycle must try to break them)
- does `fingerprint` survive as a public concept or fuse into run()?
- `tries`/`workers` defaults: named constants or noise?