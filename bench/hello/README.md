# hello — the /sentiment benchmark

`benchmark.json` is the whole exam — task (in: text, out: pos|neg) + 6 cases —
pure data, locatable by ontology path /sentiment. Scoring is exact match
(the engine's built-in); custom scoring = inject a python scorer.

`stubs.py` holds two example exam-takers (systems under test). They are not
part of the benchmark; any invoked program can take the exam.

    .venv/bin/python -m pytest nb_tests -q                  # the acceptance run
    .venv/bin/python -m nb bench /sentiment good            # run one taker, write artifact
    .venv/bin/python -m nb bench /sentiment dumb