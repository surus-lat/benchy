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

    The SYSTEM pillar. One kind: 'keyword' (pos words + default for no match).
    A constant system is the degenerate case: pos=[], default=value. Cloud
    kinds join here as specs, not code — serving is long-term work; the
    compiler is the only thing that grows.
    """
    if spec["kind"] != "keyword":
        raise ValueError(f"unknown system kind: {spec['kind']!r}")
    words = [w.lower() for w in spec["pos"]]
    default = spec["default"]

    def invoke(text: str) -> str:
        return "pos" if any(w in text.lower() for w in words) else default

    return invoke


def grade(benchmark: dict, invoke: Callable[[str], str]) -> dict:
    """Grade the exam: every case taken, scored, aggregated. Returns artifact.

    Scoring is fused here: the only rule is 'match' (1pt per exact match,
    exam score = mean). Unknown rules/aggregates are refused, not guessed.
    """
    if benchmark["scoring"]["rule"] != "match":
        raise ValueError(f"unknown scoring rule: {benchmark['scoring']['rule']!r}")
    if benchmark["scoring"]["aggregate"] != "mean":
        raise ValueError(f"unknown aggregate: {benchmark['scoring']['aggregate']!r}")
    rows = []
    for i, case in enumerate(benchmark["cases"]):
        prediction = invoke(case["input"])
        expected = case["expected"]
        rows.append({
            "case": i,
            "input": case["input"],
            "expected": expected,
            "prediction": prediction,
            "score": 1.0 if prediction == expected else 0.0,
        })
    return {
        "benchmark": benchmark["path"],
        "cases": rows,
        "score": sum(r["score"] for r in rows) / len(rows),
    }


def run(benchmark: dict, system: dict) -> dict:
    """Take the exam: system is the ARGUMENT, not a constructor field."""
    return grade(benchmark, compile(system))


def as_loss(benchmark: dict) -> Callable[[dict], float]:
    """Export the benchmark as a loss over systems: loss(dumb) > loss(good)."""
    def loss(system: dict) -> float:
        return 1.0 - run(benchmark, system)["score"]

    return loss