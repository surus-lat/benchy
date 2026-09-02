"""the engine: five functions over the four data pillars.

load     DATA pillar        bench_root/<path>.json -> benchmark dict
compile  SYSTEM pillar      spec dict -> invoke(text)->pred (the compiler)
grade    SCORING pillar     benchmark + predictions -> artifact
run      exam               benchmark + system -> artifact (system = argument)
as_loss  export             benchmark -> (system) -> float
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def load(bench_root: str, path: str) -> dict:
    """Load a benchmark by ontology path, e.g. load(root, '/sentiment')."""
    file = Path(bench_root) / (path.strip("/") + ".json")
    return json.loads(file.read_text(encoding="utf-8"))


def compile(spec: dict) -> Callable[[str], str]:
    """Compile a system spec into invoke(text) -> prediction.

    The SYSTEM pillar. Today: deterministic offline kinds (keyword, constant).
    Cloud kinds (openai-compatible endpoints) join here as specs, not code —
    serving is long-term work; the compiler is the only thing that grows.
    """
    kind = spec["kind"]
    if kind == "keyword":
        words = [w.lower() for w in spec["pos"]]
        default = spec["default"]

        def invoke(text: str) -> str:
            return "pos" if any(w in text.lower() for w in words) else default

        return invoke

    if kind == "constant":
        return lambda text: spec["value"]

    raise ValueError(f"unknown system kind: {kind!r}")


def _score_one(rule: str, expected: str, prediction: str) -> float:
    """One scoring rule. 'match' = exact match, 1pt; 'contains' = substring."""
    if rule == "match":
        return 1.0 if prediction == expected else 0.0
    if rule == "contains":
        return 1.0 if expected in prediction else 0.0
    raise ValueError(f"unknown scoring rule: {rule!r}")


def grade(benchmark: dict, invoke: Callable[[str], str]) -> dict:
    """Grade the exam: every case taken, scored, aggregated. Returns artifact."""
    rule = benchmark["scoring"]["rule"]
    rows = []
    for i, case in enumerate(benchmark["cases"]):
        prediction = invoke(case["input"])
        expected = case["expected"]
        rows.append({
            "case": i,
            "input": case["input"],
            "expected": expected,
            "prediction": prediction,
            "score": _score_one(rule, expected, prediction),
        })
    agg = benchmark["scoring"]["aggregate"]
    if agg != "mean":
        raise ValueError(f"unknown aggregate: {agg!r}")
    artifact = {
        "benchmark": benchmark["path"],
        "cases": rows,
        "score": sum(r["score"] for r in rows) / len(rows),
    }
    return artifact


def run(benchmark: dict, system: dict) -> dict:
    """Take the exam: system is the ARGUMENT, not a constructor field."""
    return grade(benchmark, compile(system))


def as_loss(benchmark: dict) -> Callable[[dict], float]:
    """Export the benchmark as a loss over systems: loss(dumb) > loss(good)."""
    def loss(system: dict) -> float:
        return 1.0 - run(benchmark, system)["score"]

    return loss