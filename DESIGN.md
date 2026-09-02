# loss-first benchy — DESIGN

## the identity

```
benchmark = (Task, Data, Scoring)          # all three are DATA
loss      = benchmark.as_loss()            # (System) -> float
receipt   = benchmark.run(system)          # the evidence trace of ONE loss eval
```

The benchmark IS a loss function over systems. `as_loss()` is not a feature,
it is the identity. `run(system)` is not a runner concept — it is the receipt
projection: keep the trace of the one evaluation that produced the float.
CLI, artifacts, grading: projections of `(Task, Data, Scoring, System) -> float`.

## shapes

- benchmark = `bench.json` (data): `path` (ontology /<task>/<domain>/<lang>),
  `task` (in/out schema), `scoring` (compare + aggregate), `cases`.
- system = a program: any importable `solve` callable. `solve(input) -> prediction`.
  Model, node, workflow, agent — same thing: it is invoked, it predicts.
- receipt = dict: path, score, aggregate, per-case [{i, in, want, got, score}].

## concept table

| concept | pillar | why undeletable | survived N |
|---|---|---|---|
| load | DATA | data must enter somehow; the only IO concept | 0 |
| Bench | TASK | the closure of task+data+scoring as one callable value | 0 |
| Bench.run | SCORING | the single evaluation; produces receipt (trace + float) | 1 |
| Bench.as_loss | SCORING | THE identity; deleting it deletes the angle | 0 |
| system | SYSTEM | systems must enter as programs; only loader concept | 0 |
| SCORES/AGGS | SCORING | data-declared scoring needs an interpretation table | 1 |
| ROOT | TASK | ontology path -> file must anchor somewhere | 0 |

## what broke and what it proved

- cycle 1: `aggregate` key in receipt — test failed → the key is derivable from
  `path` -> spec; receipt carries results, not schema. Noise removed (fixed tests forward).
- cycle 1: SCORES/AGGS extra vocab (`fuzzy`, `sum`, `min`) — nothing used them; deleted. No break.