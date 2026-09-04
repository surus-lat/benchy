BARE-METAL GOLEM REPORT
========================
Iteration: 7
Worktree: all (main + 8 worktrees) + throwaway composition
Files scanned: benchy/core.py (9 identical copies), benchy/__init__.py,
               tests/benchy/test_core_contracts.py, 5 module trees

VIOLATIONS:
- [CRITICAL→FIXED] benchy/core.py: Capabilities carried `kind`/`SystemKind`,
  `tools`, `streaming`, `batch`, `video_in`, `context_tokens` — writers
  without readers (grep-exhaustive across all 9 trees: 9 re-exports,
  4 writers, 0 readers for kind; 0 readers for the rest). Superstition.
  CUT in iteration 7.
- [CRITICAL→FIXED] benchy/core.py: `Data.split` — 0 callers anywhere;
  split selection lives at the source level (`jsonl:`, `hf:`). CUT. (wt/data
  keeps its concrete `split()` as an implementation convenience; the
  protocol no longer demands it.)
- [WARNING] .worktrees/engine/benchy/benchmark.py: `ckpt.clear()` ran
  unconditionally — an errored run destroyed its own checkpoint, defeating
  resume. Latent bug, pre-existing my cut, surfaced by the push's full-suite
  run. FIXED: clear only when n_errors == 0.
- [WARNING] .worktrees/data/benchy/data/dataset.py: `__len__` on a
  length-unknown lazy source silently walked the entire source (and
  `list()` triggered it via CPython preallocation → hidden double pass:
  400k rows walked for a 200k source). FIXED: `__len__` raises TypeError
  when unknown; `sample(n)` now passes `len_hint=n` (the reservoir size
  is knowable); sources that know their length still answer cheaply.
- [WARNING] .worktrees/data: `benchy.data` did not export `load` — the
  seam contract the worktree agents were handed. FIXED: re-exported.
- [NITPICK] wt/scoring and wt/data carry an in-flight rewrite from a
  parallel session (untracked files + stash@{0} in wt/scoring). Its
  mid-refactor state (bare-function scoring, `Data` unexported) was
  internally inconsistent independent of this push; left parked, not
  judged — that session owns it. The committed scoring tree passes 271
  tests against the cut spine.

BREAKAGE STATUS: BROKE (as required)
- wt/system: 50 test failures — TypeError on `Capabilities(kind=...)` at
  4 source sites + 3 test asserts. Repaired: the kind tag is gone because
  nothing branches on it (the theorem, not a workaround).
- wt/engine: 1 failure — checkpoint cleared after errored run (pre-existing
  latent bug, surfaced). Repaired at root.
- wt/data: 1 failure — hidden double pass in `list(sample())`. Repaired at
  root (CPython length-hint semantics).
- wt/scoring, wt/task: no failures against the cut (scoring's collection
  error was the parallel session's uncommitted WIP, orthogonal).

CUT EVIDENCE (grep across all 9 trees, this session):
- SystemKind/kind: 9 __init__ re-exports, 4 Capabilities(kind=...) writers,
  0 readers → CUT. test locks updated in all 9 copies.
- tools/streaming/batch/video_in/context_tokens: 0 readers → CUT.
- Data.split: 0 callers → CUT from protocol (concrete impl retained in
  wt/data as source-level convenience).
- RETAINED with citations: aclose (c: resource-release wire; caller-owned
  lifecycle — engine never calls it), Response.raw (c: provider payload
  for custom parse_fn), Score.scorer (b: report), Sample.meta (c: choices
  to scorers), Record.raw_text (b: report/audit), Usage fields (b: report),
  Capabilities 5 fields (a+b: accepts()/render/parse branches + run-loop
  consumer), Request.params/meta (b: system lowering + echo correlation).

EXHAUSTION PROOF: test_the_public_surface_census + Capabilities field-set
lock added to tests/benchy/test_core_contracts.py (propagated to all 9
trees). Every remaining spine name is enumerated and must cite (a)/(b)/(c)
— see .plans/BARE-METAL-THEORY.md §4–5. The census test makes drift a
test failure, not a debate.

SEAM GATE: composed all five modules in /tmp/benchy-compose against the
CUT spine — 50/50 passed (27 spine + 23 seam). First time the merge gate
has ever run green. Round 2 (integration) is proven achievable; the seam
suite's `benchy.data.load` expectation is now satisfied by wt/data.

VERDICT: PASS

The spine's surface is cut — Capabilities 11 fields → 5, Data protocol
4 methods → 3, one alias type gone — while the line count held at 425
because every cut was replaced by its evidence docstring. Surface, not
lines, is what the golem counts. The
push broke three worktrees and surfaced two latent bugs (checkpoint
self-destruction; hidden double pass) — both fixed at root, both now
locked by their tests. The golem spec itself was cut in the same motion:
it guarded a ghost primitive (`AISystem.__call__` from the superseded
plan); it now guards the exam algebra and cites the Cutting Theorem.

Open items (next cycles, not this one):
- The estimator clause of the theory (mean/min/quantile point estimates)
  is currently satisfied by "override aggregate()"; if a run of real
  benchmarks needs a non-mean estimator at the CLI level, that is the
  next name to add — deliberately, with a cited branch.
- Round 2 proper: land the composition into the main tree (the /tmp
  compose proved it merges clean).
- Round 4 nuke of src/ still gated on explicit approval.