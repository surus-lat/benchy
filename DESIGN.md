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

## queued deletion candidates (loudest first)

1. `ROOT` — filesystem anchor; could load() take a relative glob from CWD instead?
2. `loss.trace` — is the receipt as a function attribute the right home? could
   the loss RETURN (float, trace)? Bar says loss must be (System)->float, so
   attribute is forced. Verify convention is bare metal.
3. `1.0 - score` loss convention — acceptance bar demands loss(dumb) > loss(good);
   lower=better is forced by the bar. Bare metal by fiat.
4. `system()` concept — could systems be plain importables the caller passes?
   The exec-loading is a convenience; try making the test call the stubs directly.