"""nb.exam — the exam: benchmark = data + scoring; system is the argument.

The only behavior in the engine: a for-loop over cases, a mean, and JSON I/O.
"""
import json
from pathlib import Path
from typing import Callable



def exact_match(case, prediction) -> float:
    """The dumbest Scorer: 1 point per exact match."""
    return float(prediction == case["expected"])


SCORINGS = {"exact_match": exact_match}
# data-declared scoring kinds; a scorer is any callable (case, prediction) -> float


class Exam:
    """One benchmark.  run(system) grades a taker; as_loss() exports the loss."""

    def __init__(self, cases, scorer, path: str = ""):
        self.cases, self.scorer, self.path = cases, scorer, path

    def run(self, system) -> dict:
        """The system takes the exam; returns the graded artifact (JSON-ready)."""
        pages = []
        for case in self.cases:
            prediction = system.invoke(case["input"])
            pages.append({**case, "prediction": prediction,
                          "score": self.scorer(case, prediction)})
        score = sum(p["score"] for p in pages) / len(pages)
        return {"benchmark": self.path, "cases": pages,
                "score": score, "loss": 1.0 - score}

    def as_loss(self) -> Callable:
        """The benchmark as a loss function over systems: lower is better."""
        def loss(system) -> float:
            return float(self.run(system)["loss"])

        return loss


def load(bench_dir: str | Path) -> Exam:
    """Read a benchmark directory — benchmark.json is the whole exam, pure data."""
    data = json.loads((Path(bench_dir) / "benchmark.json").read_text())
    return Exam(data["cases"], SCORINGS[data["scoring"]["kind"]], data["path"])


def locate(bench_root: str | Path, path: str) -> Exam:
    """Find a benchmark by ontology path /<task?>/<domain?>/<language?>."""
    for f in sorted(Path(bench_root).rglob("benchmark.json")):
        if json.loads(f.read_text())["path"] == path:
            return load(f.parent)
    raise LookupError(f"no benchmark with ontology path {path!r}")