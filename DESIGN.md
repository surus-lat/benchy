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
| Bench.evaluate | SCORING | the single loss evaluation; the identity's engine room | 0 |
| Bench.as_loss | SCORING | THE identity; deleting it deletes the angle | 0 |
| Bench.run | DATA | receipt projection of one loss eval; the artifact source | 0 |
| system | SYSTEM | systems must enter as programs; only loader concept | 0 |
| Bench | TASK | the closure of task+data+scoring as one callable value | 0 |
| SCORES/AGGS | SCORING | data-declared scoring needs an interpretation table | 0 |
| ROOT | TASK | ontology path -> file must anchor somewhere | 0 |

## push plan (the loudest things first)

1. `load`'s filesystem fallback — ontology path is the contract; does FS search need to exist?
2. SCORES/AGGS tables — can scoring be data-computed instead of table-looked?
3. Bench as class — is a class needed, or is the benchmark a plain function?
4. `run` vs `evaluate` — fuse them.