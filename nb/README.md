# nb — a benchmark is a directory

A benchmark is a DIRECTORY of data files. The engine (`nb/`) is a pure
interpreter of those directories. Zero Python to define, run, or score a
benchmark. Python is the escape hatch, never the interface.

## The directory format (the schema)

A benchmark lives under `bench/<name>/` and is located by ontology path:
`/<task?>/<domain?>/<language?>` — the benchmark's declared `task:` is what
makes it locatable at `/sentiment`.

```
bench/<name>/
  task.yaml        REQUIRED  task name (the ontology path segment) + in/out types
  scoring.yaml     REQUIRED  how to grade a prediction against expected
  cases.jsonl      REQUIRED  the exam: one case per line
  systems/         OPTIONAL  zero or more systems, each one data too
    good.json      ...
    dumb.json      ...
```

Design note (cycle 0): stdlib has NO yaml parser. Writing one is a big
concept (indentation state machine, scalars/anchors/flow styles). JSON
`json` is stdlib and non-engineers write it fine. Per the design-freedom
note in the brief, the format drops to JSON — the engine stays a pure
directory interpreter either way. The yaml hypothesis is testable without
the yaml machinery: the METAL is "benchmark = directory of data files";
the file extension is detail. Trade-off recorded in LEARNINGS.md.

```
bench/<name>/
  task.json        REQUIRED  {"task": "sentiment", "in": "text", "out": "label[pos|neg]"}
  scoring.json     REQUIRED  {"match": "exact", "points": 1, "aggregate": "mean"}
  cases.jsonl      REQUIRED  {"input": "...", "expected": "pos"} per line
  systems/good.json, systems/dumb.json
```

## The four pillars map to files

- TASK → task.json
- SCORING → scoring.json
- DATA → cases.jsonl
- SYSTEM → systems/*.json

## Usage

```
python3 -m nb.run bench/hello systems/good
```

prints a graded artifact JSON to stdout (per-case scores + aggregate) and
writes it to `runs/<bench>/<system>/<ts>.json`. `as_loss()` = 1 - score;
loss(dumb) > loss(good).