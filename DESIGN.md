# s05 design — a benchmark is a directory

Hypothesis: a benchmark is a DIRECTORY of data files; the engine is a pure
interpreter of directories. Zero Python to define, run, and score a benchmark.

## The shape

```
bench/<name>/
  task.json       {"task": "sentiment"} — the ontology locator
  scoring.json    one of the two scoring literals (below) — whole-dict loud check, unknown keys raise
  cases.jsonl     {"input": ..., "expected": ...} per line
  systems/*.json  {"kind": "constant"|"keyword", ...}
```

## The scoring vocabulary (cycle 6 — the angle's key probe, brief-mandated)

Weighted/partial scoring, expressed in pure data — no Python in the
benchmark, no escape hatch:

```
{"match": "exact"}                            all-or-nothing per case
{"match": "fields", "weights": {field: w}}    weighted partial credit:
  per-case score = sum(w of matching fields) / sum(all w)
```

Both literals are checked WHOLE-DICT loud (cycle 5 law): any other key,
any other match value, weights that do not name every expected field →
raise. The `fields` form is the vision's hierarchy-of-importance
(IDEAS.md: "one field critical, the rest nice to have") as pure data —
worked example: `bench/extract/` (weights rank a 1-field critical hit
ABOVE a 2-field nice-to-have hit). The probe also exposed `"points"` as
noise: the cycle-5 loud check had pinned it to a single possible value —
a constant masquerading as a variable — so it is deleted from the
vocabulary and from hello's scoring.json.

Engine: `nb/bench.py` — interpreter + CLI in one file
(`python3 nb/bench.py <bench_dir> <system>`). Artifact: JSON to stdout +
`runs/<bench>/<system>.json`, per-case scores + aggregate. `as_loss()` =
1 - score.

## The yaml question (cycle 0)

The angle hypothesized `task.yaml + scoring.yaml`. Stdlib has NO yaml
parser; a hand-rolled one is a big concept (indentation state machine,
anchors, flow styles) serving zero vision pillars — it would be noise
wearing the angle's clothing. Per the design-freedom note and the law
hierarchy (VISION/IDEAS > GOLEM > angle), the format drops to JSON. The
METAL — "benchmark = directory of data files, engine = pure interpreter"
— survives untouched. Trade-off recorded in LEARNINGS.md.

## Concept table

| concept | pillar | why undeletable | survived N |
|---|---|---|---|
| load | DATA | turns directory into task+scoring+cases(+system); the interpreter itself; absorbed load_system cycle 3 | 1 |
| invoke | SYSTEM | the ONLY system call: data-dict -> prediction; the AI-API | 0 |
| run | ALL | result = benchmark.run(system); the vision invariant; grade fused away cycle 2, its loop inlined cycle 6 | 0 |
| score_case | SCORING | interprets the two pure-data scoring literals (exact / fields+weights); grown cycle 6 under --allow-growth, brief-mandated by the s05 key probe; "points" deleted same cycle | 0 |
| as_loss | SCORING | loss = benchmark.as_loss(); the software-3.0 export | 0 |
| save | DATA | graded artifact persistence (runs/<bench>/<system>.json) | 0 |

Deleted so far: `main` (cycle 1, CLI noise), `grade` (cycle 2, fused into
run's loop — scoring data is interpreted inline), `_read_json`/`_read_jsonl`
(cycle 2, one-line wrappers — json.loads called directly), `load_system`
(cycle 3, fused into load as an optional `system` name param — run() now
accepts a system NAME or data dict; one loader concept, not two),
`task.json in/out keys` (cycle 4 — declared but never interpreted: unread
schema keys are noise in the DATA format too; the deletion law extends into
the benchmark files. The TASK pillar survives as the `task` name, which the
engine reads as the ontology locator and stamps into the artifact; type
documentation for authors lives in the cases themselves), `scoring.json
"aggregate": "mean"` (cycle 5 — the last unread scoring key; its deletion
was enabled by replacing the per-key read with a WHOLE-DICT loud check:
`scoring != {"match": "exact", "points": 1}` raises. Any key the engine
does not interpret now fails loudly instead of lying silently — the noise
law, enforced by the interpreter's strictness, not by documentation).