BARE-METAL GOLEM REPORT
========================
Iteration: 8
Worktree: all (main + 8 worktrees) + throwaway composition
Files scanned: benchy/core.py (9 copies), benchy/__init__.py, consumer map
               across every module tree

VIOLATIONS:
- [CRITICAL→FIXED] benchy/core.py: `Response.raw` — 5 writers (system
  adapters), 0 readers. Cycle-1's clause-(c) citation ("carrier to custom
  parse_fns") was FALSE: it existed only in the superseded plan, and the
  cycle-1 reader-regex was broken (matched writer lines). CUT.
- [CRITICAL→FIXED] benchy/core.py: `System.aclose` in the protocol —
  0 production callers across all 9 trees. Cycle-1's "resource-release
  wire" citation had topology but no traffic. CUT from the protocol;
  `BaseSystem.aclose` survives as adapter convenience (wt/system's own
  tests assert it), because the protocol must not mandate what no engine
  code calls. Lifecycle belongs to the system's owner.
- [CRITICAL→FIXED] benchy/core.py: `ParseFailure` — 0 raisers, 0 catchers.
  Worse: it contradicts the landed contract — `task/test_parse_never_raises.py`
  locks "parse never raises; failure flows through `Prediction.parse_ok=False`".
  A fossil of the superseded raise-on-parse design. CUT.
- [WARNING] tests/benchy/test_core_contracts.py: cycle-1's census lock and
  duck-tests survived by requiring aclose on the duck — the lock itself
  enforced superstition. Fixed: new locks (Response field-set lock,
  aclose-free System duck) make the cut test-enforced.
- [WARNING] THEOREM SHARPENED: clause (c) rewritten from "minimal wire" to
  "TRAFFIC": a wire is proven by traffic, not topology — something at the
  far end must read it today. Evidence classes now rank:
  production reader > production writer > test-only anything. Readers
  imply their writers; writers imply nothing. (Both .plans docs updated.)
- [NITPICK] my own cycle-1 reader-map regex used unsupported lookahead
  (`(?!...)` in grep -E) and produced false zeros/false readers — the map
  is only as good as its grep. Cycle-2's map re-ran every suspicious name
  with hand-verified reader/writer discrimination.

BREAKAGE STATUS: BROKE (as required)
- wt/system: 9 failures, all in the predicted class: `Response(raw=...)`
  writers (openai_system ×2, hf/canary, hf/whisper, hf/voxtral) plus the
  endpoint-parity test that exercises openai's lowering. Repaired by
  deleting the kwarg — the souvenir is gone, the adapters are lighter.
- No other tree moved: scoring 273 (+2 new spine locks), task 176 (+2),
  data 45 (+2), engine 64 (+2), mirrors 29 (+2). The cut was surgical:
  every other module tree was already honest about not reading .raw.

CUT EVIDENCE (grep-exhaustive, cycle-2 map):
- Response.raw: writers openai_system.py:269,292; hf/canary.py:117;
  hf/whisper.py:110; hf/voxtral.py:118. Readers: NONE (verified by
  hand-checked grep for `.raw` outside `raw_text` and local-variable
  `raw =` assignments).
- System.aclose: production callers NONE across 9 trees. Test callers:
  test_echo, test_openai_system, test_hf_whisper, test_python_loader,
  engine/conftest.py fixtures (asserting concrete adapter behavior, not
  protocol membership).
- ParseFailure: 9 × re-export lines in `__init__.py` + 9 × census lock
  lines. Zero raisers. Zero catchers.

EXHAUSTION PROOF (remaining spine names, all cite (a)/(b)/(c)):
- Scorer.fitness: RETAINED — real production consumer found: loss.py
  `as_metric` calls `scoring.fitness(prediction, expected, sample)`.
  (Cycle-2's initial hypothesis that it was a ghost was FALSIFIED by the
  map. The map corrected the theorist, not the code.)
- OntologyPath.parse/is_prefix_of/segments: consumers task/registry.py,
  benchmark.py, cli.py. Score.scorer: report.py:73. Response.ok: run loop
  (benchmark.py:121). Sample.meta: scoring primitives + data cache + run
  loop. Prediction.parse_ok/parse_error/raw_text: report.py + seam tests
  + task tests. Usage/latency_ms: report.py + adapters. Request.params:
  openai_system + hf/whisper. Message.text: 6 files. max_concurrency:
  run loop (benchmark.py:254). accepts: task/base.py. capabilities:
  run loop + task bridge + seam tests. SystemLoader: system/registry.py.
  LossFn: loss.py + benchmark.py + __init__. All errors that remain
  (LoadError, SchemaViolation, SystemFailure, CapabilityError) have
  raisers AND catchers in production code.

SEAM GATE: recomposed all five modules against the cycle-2 spine in
/tmp/benchy-compose: 54/54 (29 spine locks + 25 seams). The merge gate
stays green through the cut. NEW this cycle: Seam 1b — the composition
claim ("any workflow is just another System; the host language is the
composition algebra; python: is the universal constructor") is now an
executable lock: a two-stage segment→extract→merge workflow written in
host Python, loaded via python:, graded by the real engine at fitness 1.0,
indistinguishable from a raw model. Also locked: the lazy dspy/textgrad
adapters raise clean actionable ImportErrors when the frameworks are
absent (wt/engine +2).

VERDICT: PASS

Cycle 2's real product is not the three names it cut — it is the
correction of cycle 1's proof. A citation standard that counts writers as
readers, or topology as traffic, will always converge to "keep
everything"; the theorem now forbids both moves. The spine is at 425
lines with Response 6 fields → 5, System protocol 3 members → 2, errors
6 → 5. Three names left the spine and the golem's own reports 007 lost
three false citations. The next cycle's open questions, honestly filed:
- Is `Report.meta` read anywhere? (unmapped this cycle)
- Is `Record.raw_text` read outside report.py? (mapped: report.py:2 —
  thin but real)
- The two CLI surfaces (engine/benchy/cli.py vs wt/cli/benchy/cli/eval.py)
  still disagree; Round-2 must pick one and delete the other.

DELIBERATELY DEFERRED: registry of compilers (progressive), scoring
primitives library (the algebra is the product), src/ nuke (Round 4,
gated), estimator surface (encode g() when a real benchmark needs it).