"""nb — the benchy redesign, loss-first.

The identity: a benchmark IS a loss function over systems.

    loss = load("/sentiment")               # (System) -> float
    loss(system)                            # one evaluation -> float
    loss.trace                              # the receipt: that eval's evidence

Everything else — grading, artifacts, CLI — is a projection of
(Task, Data, Scoring, System) -> float.

A benchmark is DATA (bench.json), locatable by ontology path /<task>/<domain>/<lang>.
A system is a program: any importable `solve` callable. It is invoked; it
produces a prediction. Model, node, workflow, agent — all the same thing here.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    """Load a benchmark by ontology path (/sentiment). Benchmarks are data;
    the path is the identity. No second addressing scheme. Returns the loss
    itself: (System) -> float, with the receipt at loss.trace."""
    for f in ROOT.glob("bench/**/bench.json"):
        spec = json.loads(f.read_text())
        if spec["path"] != path:
            continue

        def loss(system) -> float:
            if spec["scoring"] != {"compare": "exact", "aggregate": "mean"}:
                raise LookupError(f"unknown scoring: {spec['scoring']}")
            cases = []
            for i, case in enumerate(spec.get("cases", [])):
                got = system(case["in"])
                score = float(got == case["want"])
                cases.append({"i": i, "in": case["in"], "want": case["want"],
                              "got": got, "score": score})
            loss.trace = {"path": spec["path"],
                          "score": sum(c["score"] for c in cases) / len(cases),
                          "cases": cases}
            return 1.0 - loss.trace["score"]

        return loss
    raise LookupError(f"no benchmark with ontology path {path!r}")


def system(name):
    """Load a system program by file path: `bench/hello/systems/good.py`.
    A system is its file — no second addressing scheme."""
    p = Path(name)
    ns = {}
    exec(compile(p.read_text(), str(p), "exec"), ns)
    return ns["solve"]