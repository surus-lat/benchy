"""nb.cli — the product is two commands (the whole UX):

    benchy run <bench> <system> [--limit N]   take the exam, write the graded run
    benchy new <name>                          scaffold a benchmark

The graded artifact IS the report — JSON, indent=1, per-case input/expected/
prediction/score + the aggregate. It interprets alone (no engine needed to
read your evidence); a digest verb would be a formatter, not a product word.

`benchy` is `python -m nb` until packaging earns a console script.
bench/ is the exam corpus (addressed by ontology path), runs/ the evidence.
"""
import importlib.util
import json
import sys
from pathlib import Path

from .exam import locate

EXAM = '{"task": {"input": "", "output": ""}, "cases": []}\n'
# the scaffold is pure data — a benchmark is data; the ontology path is the
# directory itself (cycle 6 killed the `path` field — a second address);
# the taker (systems.py) is yours to write next to the exam, where run finds it


def run(bench, system, limit=None):
    """`run <bench> <system> [--limit N]`: the system takes the exam (first N
    cases if limited — the smoke valve), the graded artifact lands in runs/."""
    exam = locate("bench", bench)
    if limit:
        exam.cases = exam.cases[:limit]
    d = Path("bench") / bench.lstrip("/")
    spec = importlib.util.spec_from_file_location("systems", d / "systems.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, system):
        raise LookupError(f"no taker {system!r} in {d}/systems.py")
    # the artifact interprets alone (c4) so it carries its own identity —
    # WHO took / WHICH exam — composed here, not in Exam: grading is pure
    # (cases + score); identity belongs to the write, and the CLI already
    # holds both addresses.  Exam.path/Exam.dir died as lenses (cycle 7).
    artifact = {"system": system, "benchmark": bench, **exam.run(getattr(mod, system))}
    out = Path("runs") / f"{bench.lstrip('/').replace('/', '-')}-{system}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"{bench} · {system}: score={artifact['score']:.2f} "
          f"loss={1.0 - artifact['score']:.2f} -> {out}")


def new(name):
    """`new <name>`: scaffold the exam as pure data — creating exams is the
    product's first step, so the format starts as a written file, not lore."""
    d = Path("bench") / name
    if (d / "benchmark.json").exists():
        raise ValueError(f"{d}/benchmark.json already exists")
    d.mkdir(parents=True, exist_ok=True)
    (d / "benchmark.json").write_text(EXAM)
    print(f"{d}/  — add cases to benchmark.json, write systems.py, then: run /{name} <taker>")
