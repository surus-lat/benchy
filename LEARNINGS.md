# s05 learnings — what the bare metal actually is

(finished at the stop condition — 15 cycles, golem --final PASS)

## The angle's key probe — cycle 6: did benchmark-as-directory survive?

**YES — weighted/partial scoring is pure data; the escape-hatch cost so
far is ZERO.** The vision's hierarchy-of-importance (IDEAS.md: "one field
critical, the rest nice to have") is expressible as one JSON literal:

    {"match": "fields", "weights": {"total": 5, "vendor": 1, "date": 1}}

per-case score = sum(weights of matching fields) / sum(all weights).
Proven by `bench/extract/`: the system that nails only the 5-weight field
scores 5/7 and BEATS the system that nails two 1-weight fields (2/7).
No Python in the benchmark, no escape hatch invoked.

The cycle-5 whole-dict loud check did exactly what it was designed to do:
it REFUSED the new vocabulary until the interpreter honestly extended.
That refusal is the noise law working — growth had to pass
`--allow-growth "C6 probe mandated by brief"` with the DESIGN row written
first. Cost of the extension: +7 loc, +1 concept (score_case), and the
extension itself immediately exposed a deletion: `"points": 1` was a
constant pinned by its own loud check — a variable that could never
vary. Deleted from the vocabulary and from hello's scoring.json.

## Second escape-hatch probe — cycle 14: regex systems are pure data too

The `keyword` system kind subsumed into regex alternation
(`great|excelente|loved` IS the keyword list — a keyword list is just
alternation, a second kind was a concept wearing cargo). extractor.json,
169 bytes, scores 1.0 on the weighted benchmark. Zero benchmark-author
Python. System vocabulary ends at two loud-dispatched shapes:
`constant` | `regex`.

Both escape-hatch probes (scoring complexity, system complexity) closed at
ZERO interface cost: the interpreted DATA vocabulary grew honestly under
--allow-growth, the interface never did.

## Pillar by pillar (task / scoring / data / system)

- **task (program: input→output)**: the case is the raw pair — `{"input":
  ..., "want": ...}` in cases.jsonl. No Task class, no schema keys the
  engine does not interpret (in/out deleted from task.json cycle 4 —
  unread schema noise extends INTO the data format). Task semantics live
  in the artifact's `task` stamp (ontology locator, BARE_METAL c7).
- **scoring**: scoring.json is genuinely interpreted data (BARE_METAL c8):
  `exact` | `fields+weights`. score_case fused into run (c13) — scoring is
  a branch of the interpreter, not a peer concept. Named `as_loss` is the
  vision invariant (BARE_METAL c12: deletion broke the ranking tests; the
  1-line body is the minimum).
- **data (exam)**: a benchmark IS a directory — task.json (locates,
  BARE_METAL c7), scoring.json (grades, BARE_METAL c8), cases.jsonl
  (examines), systems/*.json (takes the exam). The artifact bar is stdout
  (c11: runs/*.json was an unread third projection; this tree has no
  resume requirement, so no mid-run write — contrast s04 where kill-safety
  was BARE_METAL; the artifact contract is tree-relative, resume is not).
- **system (compiler/ai-endpoint)**: systems are JSON specs loud-dispatched
  to callables (`constant` | `regex`). A cloud endpoint is one more spec
  shape — no engine change. The system is the argument; the exam-taker is
  data.

## The yaml→JSON decision (cycle 0)

No stdlib YAML parser; hand-rolling one is a large concept serving no
pillar. JSON is stdlib, and the angle's essence (benchmark-as-directory)
survives the format swap untouched: the DIRECTORY is the insight, not the
serialization.

## What the directory format actually requires

A non-engineer can write a benchmark: three data files + system specs,
every key interpreted, unknown keys raise. The engine is 72 lines of
stdlib Python, 2 files, 4 public concepts (load, invoke, run, as_loss).

## Advice for the other searchers

- Loud checks must pin the vocabulary to what the interpreter REALLY
  reads — a pinned constant (`points`) is a variable that can never vary.
- Subsume, don't add: the keyword kind died inside regex alternation.
- The artifact bar decides the write contract: no resume → stdout IS the
  artifact (c11); resume → mid-run write is metal (s04). Both are honest;
  the bar, not the engine, chooses.
- Unread projections are noise even when they look useful (runs/*.json).