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
# escape hatch — python is not the interface).
def invoke(system, inp, ctx=None):
    """invoke a system: callable | data spec (rule+default | py)."""
    if callable(system):
        return system(inp, ctx)
    if "py" in system:                         # the escape hatch
        f, fn = system["py"].rsplit(":", 1)
        spec = importlib.util.spec_from_file_location(f.replace("/", "_"), f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, fn)(inp, ctx)
    if "rule" in system or "default" in system:
        text = inp if isinstance(inp, str) else json.dumps(inp, ensure_ascii=False)
        for cls, keys in system.get("rule", {}).items():
            if any(k.lower() in text.lower() for k in keys):
                return cls
        return system.get("default")
    raise ValueError(f"unknown system spec: {sorted(system)}")


# ---------------- TASK pillar: the program description ----------------
# the task IS its spec dict: in/out schema + ont path. ok() is one expression,
# not a class — the description of the program we are searching for.


def ok(task_spec, pred):
    """is pred a valid output? (enum membership; anything else passes)"""
    enum = task_spec.get("out", {}).get("enum")
    return True if enum is None else pred in enum


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
# The graded evidence IS the artifact dict: {ont, system, score, loss, cases}.
# No Exam object — run() returns the artifact itself. The artifact is data
# end to end: written to disk unchanged, re-read by resume unchanged.


def exam(ont, system, rows):
    """rows (per-case scores) -> the graded artifact: aggregate + evidence."""
    score = sum(r["score"] for r in rows) / len(rows) if rows else 0.0
    return {"ont": ont, "system": system, "score": score,
            "loss": 1.0 - score, "cases": rows}


def _write(path, artifact):
    """atomic artifact write: a kill must never corrupt the resume evidence."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(artifact, indent=1, ensure_ascii=False))
    os.replace(tmp, path)


class Benchmark:
    """task + data + scoring, loaded as data. the system is the argument."""
    def __init__(self, spec, path):
        self.spec, self.path = spec, Path(path)
        self.task = spec["task"]
        self.scoring = spec.get("scoring", "exact")
        self.cases = spec["cases"]
        self.systems = spec.get("systems", {})

    @property
    def ont(self):
        return self.task.get("ont", "")

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
        name = system if isinstance(system, str) else getattr(system, "__name__", "system")
        if isinstance(system, str):
            system = self.systems[system]
        if isinstance(system, dict) and "py" in system:   # py paths: by the bench
            mod, fn = system["py"].rsplit(":", 1)
            system = {**system, "py": str(self.path.parent / mod) + ":" + fn}
        cases = self.cases[:limit] if limit else self.cases
        done = {}
        if out and Path(out).exists():                   # RESUME: keep graded
            done = {r["id"]: r for r in json.loads(Path(out).read_text())["cases"]}
        todo = [(i, c) for i, c in enumerate(cases) if i not in done]

        def take(ic):
            i, c = ic
            got = invoke(system, c["in"], c.get("ctx"))
            s = grade(self.scoring, c["want"], got) if ok(self.task, got) else 0.0
            return {"id": i, "in": c["in"], "want": c["want"], "got": got, "score": s}

        rows = dict(done)
        with ThreadPoolExecutor(max(1, workers)) as ex:   # 1 pool = the runner
            a = exam(self.ont, name, [])
            for r in ex.map(take, todo):                  # write as they land
                rows[r["id"]] = r
                if out:
                    a = exam(self.ont, name, [rows[i] for i in sorted(rows)])
                    _write(out, a)
        a = exam(self.ont, name, [rows[i] for i in sorted(rows)])
        if out:
            _write(out, a)
        return a

    def as_loss(self):
        """the benchmark as a loss function for software-3.0 optimizers."""
        return lambda system: self.run(system)["loss"]


# ---------------- CLI ----------------
def main(argv):
    """run <bench> <system> [--limit N] [--workers N] [--out F]
    loss <bench> <system>"""
    if len(argv) < 3 or argv[0] not in ("run", "loss"):
        print(__doc__)
        return 2
    bench = Benchmark.load(argv[1])
    kw, i = {}, 3
    while i < len(argv):                     # --k v pairs
        if argv[i].startswith("--") and i + 1 < len(argv):
            kw[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            i += 1
    if argv[0] == "loss":
        print(bench.as_loss()(argv[2]))
        return 0
    out = kw.get("out") or str(bench.path.parent / "runs" / f"{argv[2]}.json")
    exam = bench.run(argv[2],
                     limit=int(kw["limit"]) if "limit" in kw else None,
                     workers=int(kw["workers"]) if "workers" in kw else 1,
                     out=out)
    print(json.dumps(exam, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))