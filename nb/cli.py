"""nb.cli — the product is three commands (the whole UX):

    benchy run <bench> <system> [--limit N]   take the exam, write the graded run
    benchy new <name>                          scaffold a runnable benchmark
    benchy report <run>                        re-read a graded run: score + loss

`benchy` is `python -m nb` until packaging earns a console script.
bench/ is the exam corpus (addressed by ontology path), runs/ the evidence.
"""
import importlib.util
import json
import sys
from pathlib import Path
from textwrap import dedent

from .exam import locate

EXAM = '{"path": "/%s", "task": {"input": "", "output": ""}, "cases": []}\n'
# scaffold payload, indented so it reads as data, not engine code
SYSTEMS = dedent('''
    """Exam-takers for /%s — a system is any invoked program
    (model, node, workflow, agent).  invoke(input) -> prediction."""

    class Todo:
        """The scaffold taker: refuses to guess until you implement it."""

        def invoke(self, x):
            raise NotImplementedError("implement Todo.invoke, or write your own taker")

    todo = Todo()
''')


def run(bench, system, limit=None):
    """`run <bench> <system> [--limit N]`: the system takes the exam (first N
    cases if limited — the smoke valve), the graded artifact lands in runs/."""
    exam = locate("bench", bench)
    if limit:
        exam.cases = exam.cases[:limit]
    spec = importlib.util.spec_from_file_location("systems", exam.dir / "systems.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, system):
        raise LookupError(f"no taker {system!r} in {exam.dir.name}/systems.py")
    artifact = {"system": system, **exam.run(getattr(mod, system))}
    out = Path("runs") / f"{exam.dir.name}-{system}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"{artifact['benchmark']} · {system}: score={artifact['score']:.2f} "
          f"loss={1.0 - artifact['score']:.2f} -> {out}")


def new(name):
    """`new <name>`: scaffold the full runnable shape — creating exams is the
    product's first step, so the format starts as a written file, not lore."""
    d = Path("bench") / name
    if (d / "benchmark.json").exists():
        raise ValueError(f"{d}/benchmark.json already exists")
    d.mkdir(parents=True, exist_ok=True)
    (d / "benchmark.json").write_text(EXAM % name)
    (d / "systems.py").write_text(SYSTEMS % name)
    print(f"{d}/  — add cases to benchmark.json, then: run /{name} todo")


def report(run_path):
    """`report <run>`: re-read a graded run — evidence outlives the process
    that produced it (re-running a cloud system to see a grade costs money)."""
    a = json.loads(Path(run_path).read_text())
    lines = [f"{a['benchmark']} · {a['system']} · {len(a['cases'])} cases · task {a['task']}"]
    for c in a["cases"]:
        lines.append(f"  {'pass' if c['score'] else 'fail'}  {c['input']!r} "
                     f"-> {c['prediction']!r} (want {c['expected']!r})")
    lines.append(f"score {a['score']:.3f}  loss {1.0 - a['score']:.3f}")
    print("\n".join(lines))


def main(argv=None):
    """Dispatch the three commands; speak errors, not tracebacks."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv[:1] == ["run"]:
            rest, limit = argv[1:], None
            if "--limit" in rest:
                i = rest.index("--limit")
                limit = int(rest[i + 1])
                del rest[i:i + 2]
            run(rest[0], rest[1], limit)
        elif argv[:1] == ["new"]:
            new(argv[1])
        elif argv[:1] == ["report"]:
            report(argv[1])
        else:
            raise SystemExit(__doc__.strip())
    except (LookupError, ValueError, NotImplementedError, FileNotFoundError) as e:
        raise SystemExit(f"benchy: {e}")