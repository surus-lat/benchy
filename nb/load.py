"""LOAD — a benchmark is DATA, locatable by its ontology path /<task?>/<domain?>/<language?>.

A benchmark directory holds:
    task.json      {"in": ..., "out": ...}           — the TASK pillar
    scoring.json   {"mode": ..., "weights": ...}     — the SCORING pillar
    cases.json     [{"input":..., "expected":...}]   — the DATA pillar
    systems/*.json {"kind": ...}                     — SYSTEM specs (optional)

`load(path)` reads one benchmark dir and builds Benchmark. The CLI is
one thing: take the exam. No Python is ever required to define or run
a benchmark.
"""

import json
import sys
from pathlib import Path

from .benchmark import as_loss, run
from .scoring import score
from .system import compile_system


def load(path):
    """read one benchmark directory -> (task, scoring, cases)."""
    d = Path(path)
    spec = json.loads((d / "task.json").read_text(encoding="utf-8"))
    scoring = json.loads((d / "scoring.json").read_text(encoding="utf-8"))
    cases = [(c["input"], c["expected"])
             for c in json.loads((d / "cases.json").read_text(encoding="utf-8"))]
    if not cases:
        raise ValueError("an exam needs at least one case")  # the data door guards shape
    return (spec, scoring, cases)


def main(argv=None) -> int:
    """CLI: python -m nb.load bench/hello [system ...] — the exam takers take the exam."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    bench_path, *names = args
    bench = load(bench_path)
    sd = Path(bench_path) / "systems"
    systems = {p.stem: compile_system(json.loads(p.read_text(encoding="utf-8")))
               for p in sorted(sd.glob("*.json"))} if sd.is_dir() else {}
    for name in names or sorted(systems):
        if name not in systems:
            print(f"unknown system: {name!r} (have: {sorted(systems)})")
            return 2
        print(json.dumps({"system": name, **run(bench, systems[name])}, indent=2))
    return 0  # a graded exam is a success; a dumb score is data, not an error


if __name__ == "__main__":
    raise SystemExit(main())