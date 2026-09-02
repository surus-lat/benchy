# loss-first benchy — DESIGN

## the identity

```
loss    = load("/sentiment")        # the benchmark IS a loss over systems
loss(system)                        # one evaluation -> float
loss.trace                          # the receipt: that eval's evidence trace
```

The benchmark IS a loss function over systems. `load(path)` does not return a
Bench object with methods — it returns the loss itself: a closure
`(System) -> float`. The receipt is not a separate concept: it is the trace
attribute of the loss, kept from the evaluation that produced the float.
CLI, artifacts, grading: projections of `(Task, Data, Scoring, System) -> float`.

## shapes

- benchmark = `bench.json` (data): `path` (ontology /<task>/<domain>/<lang>),
  `scoring` (compare + aggregate), `cases`. No task key — the schema is visible
  in the cases themselves.
- system = any callable: `solve(input) -> prediction`. Model, node, workflow,
  agent — same thing: it is invoked, it predicts. The engine has NO loader;
  Python's import machinery is the loader. A system enters as a callable.
- loss.trace = dict: score, cases [{in, want, got, score}]. Evidence only —
  identity (the path) is the loss closure itself, not schema inside the receipt.

## scoring interpretation

There is no vocabulary table. The engine interprets exactly one scoring:
`{"compare": "exact", "aggregate": "mean"}` — anything else raises LookupError.
Scoring is data, but the data must NAME a scoring the engine actually knows;
unknown names fail loud, never silently score 0.

## concept table

| concept | pillar | why undeletable | survived N |
|---|---|---|---|
| load | DATA+SCORING | data must enter somehow; the only IO concept; returns the loss closure — one concept IS the whole engine | 1 |
| `want` in trace cases | SCORING+DATA | the artifact must be SELF-CONTAINED: without `want`, a failed case reads "wrong, but about what?" and no reader (human or software-3.0 optimizer) can interpret or learn from the trace without joining bench.json — which the artifact no longer names (path deleted, cycle 9). Derivable-from-source is not the bar; interpretable-alone is. | 1 |
| `scoring` key in bench.json + its load-time check | SCORING | scoring is a pillar of the benchmark identity (law #6). The key is the seat where the author names the policy the engine implements; the check is live honesty code — unknown or missing scoring fails loud, never silently exact-match. Survived deletion because nothing guarded the guard (cycle 11); the guard test now exists. | 1 |
| the inner `loss` closure | DATA+SCORING | not style — the only honest home for the receipt. `spec` must be captured at load time, and `loss.trace` must be PER-INSTANCE state: a module-level loss() would share one trace across every loaded benchmark (cycle 12 hoist attempt broke 8 tests). The closure IS the loss-first identity: load() returns the loss itself. | 1 |

(`benchmark` fused into load in cycle 3; SCORES/AGGS tables deleted in cycle 5;
`system` deleted in cycle 8 — the SYSTEM pillar needs zero engine code, a
system is a callable and stdlib importlib is the loader.)

## what broke and what it proved

- cycle 1: `aggregate` key in receipt — test failed → the key is derivable from
  `path` -> spec; receipt carries results, not schema. Noise removed (fixed tests forward).
- cycle 1: SCORES/AGGS extra vocab (`fuzzy`, `sum`, `min`) — nothing used them; deleted. No break.
- cycle 2: the Bench class — 9/10 tests failed on the push, fixed forward to the
  closure shape. run() was loss()+trace; as_loss() was the identity behind a method.
  Both are now the closure itself. NOISE_REMOVED.
- cycle 3: `benchmark(spec)` second entry point — deleted entirely, nothing broke:
  the raw-spec entry was never guarded, never used. HARD_PUSH.
- cycle 4: `system()` short-name glob fallback — deleted entirely, nothing broke:
  convenience addressing that no test or benchmark used. HARD_PUSH.
- cycle 5: SCORES/AGGS tables — deleted, replaced by one inline interpretation
  with a loud unknown-vocab check. First inline attempt was TOO SOFT (silent 0.0
  on unknown vocab = dishonest scoring); escalated to loud check. HARD_PUSH.
- cycle 6: receipt `i` key (order is the index) + `spec.get("cases", [])` default
  (missing cases crashed loudly anyway — the default was dead weight). Both deleted. HARD_PUSH.
- cycle 7: `task` key in bench.json — the engine never interpreted it; the schema
  is visible in the cases (in values show the input type, want values show the
  output vocab). Deleted from the data. CAVEAT recorded: cases carry the task
  only by EXAMPLE; a task-only description ("takes pdf, returns json fields")
  with no cases yet cannot be expressed — noted for unify, not re-added. HARD_PUSH.
- cycle 8: `system()` — deleted; a system is a callable, stdlib importlib is the
  loader, the SYSTEM pillar needs zero engine code. Tests fixed forward from
  nb.system to importlib: the loader was guarded noise (tests referenced it)
  but the GUARD was noise — the pillar is the protocol, not the loader. NOISE_REMOVED.
- cycle 9: `path` key in loss.trace — deleted; the receipt is evidence only
  {score, cases:[{in, want, got, score}]}. The identity of the benchmark is the
  loss closure itself (and the filename when serialized), not schema inside the
  artifact. Test that asserted the path key broke — its own invention, not the
  bar's; fixed forward. NOISE_REMOVED.
- cycle 10: `want` in trace cases — BARE_METAL, first earned badge. Deleted it;
  all 10 tests stayed green (nothing mechanically guarded it) but the artifact
  became an unanchored projection: a failed case read `{"in": ..., "got": ...,
  "score": 0.0}` — wrong, but about what? With `path` already deleted (cycle 9),
  the artifact couldn't even name the exam it belonged to. The vision says the
  artifact feeds software-3.0 optimizers; an optimizer cannot learn the target
  answer from a trace that omits it. Derivable-from-bench.json is not the bar —
  INTERPRETABLE-ALONE is. Restored.
- cycle 11: `scoring` key + engine check — BARE_METAL. The probe deleted both
  and all 10 tests stayed green: the honesty guard was UNGUARDED — a gap in
  the tests, not proof the pillar was ceremonial. Restored because: (a) law #6
  makes scoring a pillar of the benchmark identity ("task + data + scoring");
  (b) the check is live code — a benchmark naming `fuzzy` must fail loud,
  never silently exact-match (the cycle-5 lesson, now made load-time instead
  of hidden in the closure); (c) the vision promises multiple scorings
  ("hierarchy of importance between fields") — the key is the seat where the
  author names the policy. Fixed the gap: added a test that a missing or
  unknown scoring raises LookupError. The engine's loc is unchanged (35).
- cycle 12: `ROOT` — deleted (inlined into load's glob, loc -1). Escalation:
  hoist `loss()` to module level? BROKE — the closure needs `spec` at eval
  time and `loss.trace` needs PER-INSTANCE state; a module-level function
  shares one trace across every loaded benchmark. The closure is not style,
  it is the only honest home for the receipt. BARE_METAL (the inner closure);
  added a test proving two benchmarks keep independent traces.

## queued deletion candidates (loudest first)

1. `loss.trace` home — could the loss RETURN (float, trace)? Bar says loss
   must be (System)->float, so the attribute is forced by fiat. Probe anyway:
   does anything else about the trace convention survive scrutiny?
2. `1.0 - score` loss convention — acceptance bar demands loss(dumb) > loss(good);
   lower=better is forced by the bar. Bare metal by fiat. Try `score` AS the
   loss (higher=better): breaks the ranking direction the bar names.
3. `float(...)` cast on per-case score — is JSON-serializability of the trace
   a real requirement (bar: artifact JSON) or a nicety? `got == want` yields
   numpy/bool surprises in real systems; the cast is tiny armor.
4. `/` division by len(cases) — empty cases list = ZeroDivisionError. Loud
   crash is honest; is it? Probe: is empty-cases a valid benchmark?