#!/usr/bin/env python3
"""benchy — the engine, one file.

A benchmark is DATA (a bench.json: task + cases + scoring). The system is
the ARGUMENT:  exam = bench.run(system)   loss = bench.as_loss()(system).
Ontology: every benchmark declares its home path /<task>/<domain>/<language>.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# ---------------- SYSTEM pillar: the AI-API ----------------
# A system is a program: invoked with a case, it predicts. Model, node,
# workflow, agent — all plug in here through ONE protocol: f(in, ctx) -> out.
# Data shapes: {"rule": {cls: [keywords]}, "default": cls} (a learned keyword
# program; default-only = a constant system) and {"py": "file.py:func"} (the
# escape hatch — python is not the interface). py paths resolve from the
# CWD, same as every other data file you point at by path; run() compiles a
# py spec to a callable ONCE per exam, never per case.
def invoke(system, inp, ctx=None):
    """invoke a system: callable | data spec (rule+default)."""
    if callable(system):
        return system(inp, ctx)
    text = inp if isinstance(inp, str) else json.dumps(inp, ensure_ascii=False)
    for cls, keys in system.get("rule", {}).items():
        if any(k.lower() in text.lower() for k in keys):
            return cls
    return system.get("default")


# ---------------- TASK pillar: the program description ----------------
# The task IS data: spec["task"] = in/out schema + ont path. Zero code serves
# it: an invalid output (outside out.enum) cannot match any want, so grading
# already scores it 0. The enum is the declared output space for optimizers
# reading the task — it is not a gate the grader needs.


# ---------------- SCORING pillar: what good means (and the loss) ----------------
def grade(scoring, want, got):
    """grade -> [0,1]. 'exact' or {'fields': {name: weight}} (importance)."""
    if scoring == "exact":
        return float(want == got)
    if not isinstance(got, dict):              # fields scoring needs a dict out
        return 0.0
    w = {k: v for k, v in scoring["fields"].items() if k in want}
    if not w:
        return 0.0
    return sum(v for k, v in w.items() if want.get(k) == got.get(k)) / sum(w.values())


# ---------------- DATA pillar: the exam, taken ----------------
# The graded evidence IS the artifact dict: {ont, score, loss, cases}.
# No Exam object — run() returns the artifact itself. The artifact is data
# end to end: written to disk unchanged, re-read by resume unchanged.


def exam(ont, rows):
    """rows (per-case scores) -> the graded artifact: aggregate + evidence."""
    score = sum(r["score"] for r in rows) / len(rows) if rows else 0.0
    return {"ont": ont, "score": score, "loss": 1.0 - score, "cases": rows}


def _write(path, artifact):
    """atomic artifact write: a kill must never corrupt the resume evidence."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(artifact, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


class Benchmark:
    """task + data + scoring, loaded as data. the system is the argument.
    The engine keeps only what it uses; the task spec itself stays in the
    bench.json — pure data for optimizers, never object attributes."""
    def __init__(self, spec, path):
        self.path = Path(path)
        self.ont = spec["task"].get("ont", "")
        self.scoring = spec.get("scoring", "exact")
        self.cases = spec["cases"]
        self.systems = spec.get("systems", {})

    @classmethod
    def load(cls, ref):
        """by dir, by bench.json path, or by ontology path (/sentiment)."""
        p = Path(ref)
        f = p / "bench.json" if p.is_dir() else p
        if f.is_file():
            return cls(json.loads(f.read_text()), f)
        for f in Path("bench").rglob("bench.json"):   # ontology registry walk
            b = cls(json.loads(f.read_text()), f)
            if b.ont == ref:
                return b
        raise FileNotFoundError(f"no benchmark at {ref}")

    def run(self, system, limit=None, workers=1, out=None):
        """exam = bench.run(system). resume-safe (out), concurrent (workers)."""
        if isinstance(system, str):
            system = self.systems[system]
        if isinstance(system, dict) and "py" in system:   # compile ONCE per exam
            f, fn = system["py"].rsplit(":", 1)
            spec = importlib.util.spec_from_file_location(f.replace("/", "_"), f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            system = getattr(mod, fn)
        cases = self.cases[:limit] if limit else self.cases
        done = {}
        if out and Path(out).exists():                   # RESUME: keep graded
            done = {r["id"]: r for r in json.loads(Path(out).read_text())["cases"]}
        todo = [(i, c) for i, c in enumerate(cases) if i not in done]

        def take(ic):
            i, c = ic
            got = invoke(system, c["in"], c.get("ctx"))
            s = grade(self.scoring, c["want"], got)
            return {"id": i, "in": c["in"], "want": c["want"], "got": got, "score": s}

        rows = dict(done)
        with ThreadPoolExecutor(max(1, workers)) as ex:   # 1 pool = the runner
            for r in ex.map(take, todo):                  # write as they land
                rows[r["id"]] = r
                if out:
                    _write(out, exam(self.ont, [rows[i] for i in sorted(rows)]))
        return exam(self.ont, [rows[i] for i in sorted(rows)])

    def as_loss(self):
        """the benchmark as a loss function for software-3.0 optimizers."""
        return lambda system: self.run(system)["loss"]


# ---------------- CLI ----------------
def main(argv):
    """run <bench> <system> [out] · loss <bench> <system>
    limit/workers are engine kwargs (bench.run), not CLI flags."""
    if len(argv) < 3 or argv[0] not in ("run", "loss"):
        print(__doc__)
        return 2
    bench = Benchmark.load(argv[1])
    if argv[0] == "loss":
        print(bench.as_loss()(argv[2]))
        return 0
    out = argv[3] if len(argv) > 3 else str(bench.path.parent / "runs" / f"{argv[2]}.json")
    print(json.dumps(bench.run(argv[2], out=out), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))