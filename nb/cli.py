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

# the scaffold is pure data — a benchmark is data; the ontology path is the
# directory itself (cycle 6 killed the `path` field — a second address).
# task is a BLANK OBJECT to fill, never a typed example: any shape we print
# teaches a type the engine does not check (cycle 9 — the scaffold must not
# lie); the taker (systems.py) is yours to write next to the exam.
EXAM = '{"task": {}, "cases": []}\n'

def run(bench, system, limit=None):
    """`run <bench> <system> [--limit N]`: the system takes the exam (first N
    cases if limited — the smoke valve), the graded artifact lands in runs/."""
    exam = locate("bench", bench)
    # the artifact names its own scope (c10): total is captured BEFORE the
    # slice so a smoke run cannot masquerade as a full run — the JSON says
    # 4-of-6 without leaning on benchmark.json to interpret.  The if-limit
    # conditional died: cases[:None] IS the full exam, and --limit 0 now
    # grades nothing and says so loudly (mean of nothing) instead of
    # silently running the whole exam (the old falsy-check's lie).
    total = len(exam.cases)
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
    artifact = {"system": system, "benchmark": bench, "total": total, **exam.run(getattr(mod, system))}
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
    print(f"{d}/  — fill task {{…}}, add cases [{{\"input\": …, \"expected\": …}}], "
          f"write systems.py, then: run /{name} <taker>")
