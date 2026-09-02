# s05 — yaml / benchmark-as-directory (SUMMARY for unify)

## Final metrics (golem --final PASS — report verbatim)

```json
{"cycles": 15, "verdicts": {"NOISE_REMOVED": 4, "HARD_PUSH": 8, "BARE_METAL": 3}, "files": 2, "loc": 72, "deps": 0, "concepts": 4}
```

## Final shape

- 72 loc, 4 concepts (load, invoke, run, as_loss), 2 files, 0 deps
- A benchmark is a directory: `task.json` locates, `scoring.json` grades, `cases.jsonl` examines, `systems/*.json` takes the exam
- Scoring vocabulary: `exact` | `fields+weights` — loud-dispatched, unknown keys raise
- System vocabulary: `constant` | `regex` — specs compiled to callables; a cloud endpoint is one more spec shape
- Artifact bar: stdout (no resume here → no mid-run write; contrast s04)
- Worked examples: bench/hello (exact) + bench/extract (weighted, extractor=1.0)

## Angle status

**SURVIVED — both escape-hatch probes closed at ZERO interface cost.**
C6: weighted scoring is pure data. C14: regex systems are pure data
(keyword subsumed into alternation). The falsification never triggered;
honest growth widened the interpreted DATA vocabulary, never the interface.
BARE_METAL: task.json (ontology locator is data, c7), scoring.json (weights
genuinely interpreted, c8), as_loss (named vision invariant, c12).

## Best discovery

The deletion law extends INTO the data format: unread schema keys are noise
in data too (c4 in/out; c5 aggregate; c10 benchmark stamp). A data format
where every key is interpreted is the honest benchmark-as-directory.

## Most expensive mistake

The root `*.json` gitignore silently untracked the whole bench/ directory
for 4 cycles — until 69d5d94 negated the pattern. Verify what git tracks,
not what the tree shows.

## Advice for the other nine

The only tree where scoring vocabulary grew HONESTLY under --allow-growth —
for the unified engine: scoring complexity belongs in interpreted data
literals (weights maps, alternation), not engine branches. Loud checks make
growth honest: the interpreter refuses until it actually reads the key.

## One-line essence

A benchmark is a directory of JSON files interpreted by a 72-line engine —
task.json locates, scoring grades, cases examine, systems take the exam;
every key interpreted, unknown keys raise, loss = 1−score.