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

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORES = {"exact": lambda got, want: 1.0 if got == want else 0.0}
AGGS = {"mean": lambda xs: sum(xs) / len(xs) if xs else 0.0}


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

    def run(self, system) -> dict:
        """Receipt projection: the evidence trace of one loss evaluation."""
        cases = []
        for i, case in enumerate(self.spec.get("cases", [])):
            got = system(case["in"])
            score = SCORES[self.spec["scoring"]["compare"]](got, case["want"])
            cases.append({"i": i, "in": case["in"], "want": case["want"],
                          "got": got, "score": score})
        return {"path": self.spec["path"],
                "score": AGGS[self.spec["scoring"]["aggregate"]](
                    [c["score"] for c in cases]),
                "cases": cases}

    def as_loss(self):
        """The identity: this benchmark AS a loss function over systems."""
        def loss(system) -> float:
            return 1.0 - self.run(system)["score"]
        return loss


def system(name):
    """Load a system program by path: `bench/hello/systems/good.py` or `good`."""
    p = Path(name)
    if not p.is_file():
        p = next((ROOT / "bench").glob(f"**/systems/{name}.py"))
    ns = {}
    exec(compile(p.read_text(), str(p), "exec"), ns)
    return ns["solve"]