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

from .benchmark import Benchmark
from .scoring import score
from .system import compile_system


def load(path) -> Benchmark:
    """read one benchmark directory -> Benchmark (task+scoring+data)."""
    d = Path(path)
    spec = json.loads((d / "task.json").read_text(encoding="utf-8"))
    task = spec  # the TASK pillar is data: {"in": ..., "out": ...}
    scoring = json.loads((d / "scoring.json").read_text(encoding="utf-8"))
    cases = [(c["input"], c["expected"])
             for c in json.loads((d / "cases.json").read_text(encoding="utf-8"))]
    if not cases:
        raise ValueError("an exam needs at least one case")  # the data door guards shape
    return Benchmark(task, scoring, cases)


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
    if not names:
        names = sorted(systems)
    if not systems:
        print(f"no systems in {bench_path}/systems/")
        return 2
    for name in names:
        if name not in systems:
            print(f"unknown system: {name!r} (have: {sorted(systems)})")
            return 2
        print(json.dumps({"system": name, **bench.run(systems[name])}, indent=2))
    return 0  # a graded exam is a success; a dumb score is data, not an error


if __name__ == "__main__":
    raise SystemExit(main())