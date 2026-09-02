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
  `task` (in/out schema), `scoring` (compare + aggregate), `cases`.
- system = a program: any importable `solve` callable, addressed by FILE PATH
  only (`bench/hello/systems/good.py`). No second addressing scheme.
- loss.trace = dict: path, score, cases [{i, in, want, got, score}].

## scoring interpretation

There is no vocabulary table. The engine interprets exactly one scoring:
`{"compare": "exact", "aggregate": "mean"}` — anything else raises LookupError.
Scoring is data, but the data must NAME a scoring the engine actually knows;
unknown names fail loud, never silently score 0.

## concept table

| concept | pillar | why undeletable | survived N |
|---|---|---|---|
| load | DATA | data must enter somehow; the only IO concept; returns the loss closure | 1 |
| system | SYSTEM | systems must enter as programs; the only loader concept | 1 |
| ROOT | TASK | ontology path -> file must anchor somewhere | 0 |

(`benchmark` fused into load in cycle 3; SCORES/AGGS tables deleted in cycle 5 —
scoring is one inline interpretation, checked loud.)

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

## queued deletion candidates (loudest first)

1. receipt key minimality — `i`, `in`, `want`, `got`, `score`: is `i` needed
   when cases list is ordered? is `want` derivable from input+task?
2. `spec.get("cases", [])` default — empty benchmark = silent 0 loss; should it raise?
3. `task` key in bench.json — the schema is declared but never interpreted;
   either interpret it (validate outputs) or it is dead weight in the data.
4. `1.0 - score` loss convention — could loss be the score itself with lower=better?
   No: acceptance bar demands loss(dumb) > loss(good). Convention is bare metal.
5. `system()` as a concept — could systems just be importables the caller passes?