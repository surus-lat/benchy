# s05 learnings — what the bare metal actually is

(running notes; pillar-by-pillar summary written at the stop condition)

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

Open question for later cycles: the loud check pins the vocabulary to a
two-literal set. Every REAL scoring need now lands on score_case's
growth budget — the question is whether rubric-style or
numeric-tolerance scoring also collapses into small literals or forces
the Python escape hatch the angle feared. C8 will test whether
scoring.json itself is load-bearing now that its freedom is literal.