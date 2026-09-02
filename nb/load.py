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
from .data import Exam
from .scoring import Scoring
from .system import compile_system
from .task import Task


def _read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def load(path) -> Benchmark:
    """read one benchmark directory -> Benchmark (task+scoring+data)."""
    d = Path(path)
    spec = _read(d / "task.json")
    task = Task(spec["in"], spec["out"])  # 'in' is a keyword; read it plainly
    scoring = Scoring(**_read(d / "scoring.json"))
    exam = Exam(_read(d / "cases.json"))
    return Benchmark(task, scoring, exam)


def load_system_specs(path) -> dict:
    """read systems/*.json -> {name: spec dict} (uncompiled, for display)."""
    d = Path(path) / "systems"
    if not d.is_dir():
        return {}
    return {p.stem: _read(p) for p in sorted(d.glob("*.json"))}


def main(argv=None) -> int:
    """CLI: python -m nb.load bench/hello [system ...] — the exam takers take the exam."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    bench_path, *names = args
    bench = load(bench_path)
    specs = load_system_specs(bench_path)
    if not names:
        names = sorted(specs)
    if not specs:
        print(f"no systems in {bench_path}/systems/")
        return 2
    failures = 0
    for name in names:
        if name not in specs:
            print(f"unknown system: {name!r} (have: {sorted(specs)})")
            return 2
        artifact = bench.run(compile_system(specs[name]))
        print(json.dumps({"system": name, **artifact}, indent=2))
    return 0  # a graded exam is a success; a dumb score is data, not an error


if __name__ == "__main__":
    raise SystemExit(main())