"""nb — the benchy redesign, loss-first.

The identity: a benchmark IS a loss function over systems.

    loss = benchmark.as_loss()(system)          # (System) -> float
    receipt = benchmark.run(system)             # evidence trace of that eval

Everything else — loading, grading, artifacts, CLI — is a projection of
(Task, Data, Scoring, System) -> float.

A benchmark is DATA (bench.json), locatable by ontology path /<task>/<domain>/<lang>.
A system is a program: any importable `solve` callable. It is invoked; it
produces a prediction. Model, node, workflow, agent — all the same thing here.
"""

import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORES = {"exact": lambda got, want: 1.0 if got == want else 0.0,
           "fuzzy": lambda got, want: 1.0 if str(got).strip().lower() == str(want).strip().lower() else 0.0}
AGGS = {"mean": statistics.fmean,
        "sum": lambda xs: float(sum(xs)),
        "min": lambda xs: min(xs) if xs else 0.0}


def load(path):
    """Load a benchmark as data. path: filesystem path OR ontology path (/sentiment)."""
    p = Path(path)
    if not p.is_file():
        for f in ROOT.glob("bench/**/bench.json"):
            if json.loads(f.read_text())["path"] == str(path):
                p = f
                break
    return Bench(json.loads(p.read_text()))


class Bench:
    """A benchmark. Its whole job: evaluate systems, project the float."""
    def __init__(self, spec):
        self.spec = spec

    def evaluate(self, system) -> dict:
        """One loss evaluation of `system` over this exam. Returns the trace."""
        cases = self.spec.get("cases", [])
        scores = []
        per_case = []
        for i, case in enumerate(cases):
            got = system(case["in"])
            want = case["want"]
            score = SCORES[self.spec["scoring"]["compare"]](got, want)
            scores.append(score)
            per_case.append({"i": i, "in": case["in"], "want": want,
                             "got": got, "score": score})
        agg_name = self.spec["scoring"]["aggregate"]
        return {"path": self.spec["path"],
                "score": AGGS[agg_name](scores),
                "aggregate": agg_name,
                "cases": per_case}

    def as_loss(self):
        """The identity: this benchmark AS a loss function over systems."""
        def loss(system) -> float:
            return 1.0 - self.evaluate(system)["score"]
        return loss

    def run(self, system) -> dict:
        """The receipt projection: one loss evaluation, kept as evidence."""
        return self.evaluate(system)


def system(name):
    """Load a system program by path: `bench/hello/systems/good.py` or `good`."""
    p = Path(name)
    if not p.is_file():
        p = next((ROOT / "bench").glob(f"**/systems/{name}.py"))
    src = p.read_text()
    mod = type(sys)("sys_" + p.stem)
    mod.__dict__["__file__"] = str(p)
    exec(compile(src, str(p), "exec"), mod.__dict__)
    return mod.solve