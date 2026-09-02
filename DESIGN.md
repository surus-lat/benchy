# s05 design — a benchmark is a directory

Hypothesis: a benchmark is a DIRECTORY of data files; the engine is a pure
interpreter of directories. Zero Python to define, run, and score a benchmark.

## The shape

```
bench/<name>/
  task.json       {"task": "sentiment", "in": "text", "out": "label[pos|neg]"}
  scoring.json    {"match": "exact", "points": 1, "aggregate": "mean"}
  cases.jsonl     {"input": ..., "expected": ...} per line
  systems/*.json  {"kind": "constant"|"keyword", ...}
```

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
| run | ALL | result = benchmark.run(system); the vision invariant; scoring inlined (grade fused away cycle 2) | 0 |
| as_loss | SCORING | loss = benchmark.as_loss(); the software-3.0 export | 0 |
| save | DATA | graded artifact persistence (runs/<bench>/<system>.json) | 0 |

Deleted so far: `main` (cycle 1, CLI noise), `grade` (cycle 2, fused into
run's loop — scoring data is interpreted inline), `_read_json`/`_read_jsonl`
(cycle 2, one-line wrappers — json.loads called directly), `load_system`
(cycle 3, fused into load as an optional `system` name param — run() now
accepts a system NAME or data dict; one loader concept, not two).