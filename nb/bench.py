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
SCORES = {"exact": lambda got, want: 1.0 if got == want else 0.0}
AGGS = {"mean": lambda xs: sum(xs) / len(xs) if xs else 0.0}


def load(path):
    """Load a benchmark by ontology path (/sentiment). Benchmarks are data;
    the path is the identity. No second addressing scheme."""
    for f in ROOT.glob("bench/**/bench.json"):
        if json.loads(f.read_text())["path"] == path:
            return benchmark(json.loads(f.read_text()))
    raise LookupError(f"no benchmark with ontology path {path!r}")


def benchmark(spec):
    """A benchmark IS a loss function over systems: (System) -> float.
    loss.trace holds the receipt — the evidence of the last evaluation."""
    def loss(system) -> float:
        cases = []
        for i, case in enumerate(spec.get("cases", [])):
            got = system(case["in"])
            score = SCORES[spec["scoring"]["compare"]](got, case["want"])
            cases.append({"i": i, "in": case["in"], "want": case["want"],
                          "got": got, "score": score})
        loss.trace = {"path": spec["path"],
                      "score": AGGS[spec["scoring"]["aggregate"]](
                          [c["score"] for c in cases]),
                      "cases": cases}
        return 1.0 - loss.trace["score"]
    return loss


def system(name):
    """Load a system program by path: `bench/hello/systems/good.py` or `good`."""
    p = Path(name)
    if not p.is_file():
        p = next((ROOT / "bench").glob(f"**/systems/{name}.py"))
    ns = {}
    exec(compile(p.read_text(), str(p), "exec"), ns)
    return ns["solve"]