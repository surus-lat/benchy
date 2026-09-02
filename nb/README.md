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
  task.yaml        REQUIRED  task name (the ontology path segment)
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
  task.json        REQUIRED  {"task": "sentiment"} — the ontology path segment
  scoring.json     REQUIRED  {"match": "exact", "points": 1} — unknown keys raise
  cases.jsonl      REQUIRED  {"input": "...", "expected": "pos"} per line
  systems/good.json, systems/dumb.json
```

## The four pillars map to files

- TASK → task.json
- SCORING → scoring.json
- DATA → cases.jsonl
- SYSTEM → systems/*.json

## Usage

`python3 nb/bench.py <bench_dir> <system>` — prints the graded artifact
JSON (per-case scores + aggregate) to stdout and writes it to
`runs/<bench>/<system>.json`. `as_loss()` = 1 - score; loss(dumb) > loss(good).