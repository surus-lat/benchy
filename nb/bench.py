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


def load(path):
    """Load a benchmark by ontology path (/sentiment). Benchmarks are data;
    the path is the identity. Returns the loss itself: (System) -> float,
    with the receipt at loss.trace."""
    for f in Path(__file__).resolve().parent.parent.glob("bench/**/bench.json"):
        spec = json.loads(f.read_text())
        if spec["path"] != path:
            continue
        # scoring is data: the benchmark must NAME a policy the engine really
        # implements; unknown or missing -> loud, never silently exact-match.
        if spec.get("scoring") != {"compare": "exact", "aggregate": "mean"}:
            raise LookupError(f"unknown scoring: {spec.get('scoring')!r}")
        # the loss IS the benchmark: a per-load closure. trace must be
        # PER-INSTANCE state (two loaded benchmarks must not clobber each
        # other's receipts) — only a freshly-constructed callable gives that.
        def loss(system) -> float:
            cases = []
            for case in spec["cases"]:
                got = system(case["in"])
                cases.append({"in": case["in"], "want": case["want"],
                              "got": got, "score": float(got == case["want"])})
            loss.trace = {"score": sum(c["score"] for c in cases) / len(cases),
                          "cases": cases}
            return 1.0 - loss.trace["score"]

        return loss
    raise LookupError(f"no benchmark with ontology path {path!r}")