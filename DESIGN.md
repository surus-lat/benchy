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
- system = a program: any importable `solve` callable. `solve(input) -> prediction`.
  Model, node, workflow, agent — same thing: it is invoked, it predicts.
- loss.trace = dict: path, score, aggregate, per-case [{i, in, want, got, score}].

## concept table

| concept | pillar | why undeletable | survived N |
|---|---|---|---|
| load | DATA | data must enter somehow; the only IO concept | 0 |
| benchmark | SCORING | the loss closure: task+data+scoring fused as one callable value | 1 |
| system | SYSTEM | systems must enter as programs; only loader concept | 0 |
| SCORES/AGGS | SCORING | data-declared scoring needs an interpretation table | 1 |
| ROOT | TASK | ontology path -> file must anchor somewhere | 0 |

## what broke and what it proved

- cycle 1: `aggregate` key in receipt — test failed → the key is derivable from
  `path` -> spec; receipt carries results, not schema. Noise removed (fixed tests forward).
- cycle 1: SCORES/AGGS extra vocab (`fuzzy`, `sum`, `min`) — nothing used them; deleted. No break.
- cycle 2: the Bench class itself — load() returned an object with .run/.as_loss/.spec,
  9/10 tests failed on the push, fixed forward to the closure shape. The class was
  ceremony around the identity: run() was loss()+trace, as_loss() was the identity
  returned by a method. Both are now the closure itself. NOISE_REMOVED.

## queued deletion candidates (loudest first)

1. `system()` short-name fallback (`good` → glob `**/systems/good.py`) — convenience
   loader branch; try deleting, force explicit paths.
2. SCORES/AGGS tables vs data-computed scoring — try computing score/aggregate
   from the spec directly; the tables are an interpretation layer.
3. `benchmark()` indirection — load() could build the closure inline.
4. receipt key minimality — `i`, `in`, `want`, `got`, `score`: are all five
   needed, or derivable?