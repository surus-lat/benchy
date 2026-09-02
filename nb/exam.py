"""nb.exam — the exam: benchmark = task + data + scoring; system is the argument.

The only behavior in the engine: a for-loop over cases, a mean, and JSON I/O.
"""
import json
from pathlib import Path
from typing import Callable

from .core import Case, Scored, Scorer, System, Task


def exact_match(case: Case, prediction: object) -> float:
    """The dumbest Scorer: 1 point per exact match."""
    return float(prediction == case["expected"])


SCORINGS: dict[str, Scorer] = {"exact_match": exact_match}


class Exam:
    """One benchmark.  run(system) grades a taker; as_loss() exports the loss."""

    def __init__(self, task: Task, cases: list[Case], scorer: Scorer, path: str = ""):
        self.task, self.cases, self.scorer, self.path = task, cases, scorer, path

    def run(self, system: System) -> dict:
        """The system takes the exam; returns the graded artifact (JSON-ready)."""
        pages: list[Scored] = []
        for case in self.cases:
            prediction = system.invoke(case["input"])
            pages.append({**case, "prediction": prediction,
                          "conforms": prediction in self.task["output"]["enum"],
                          "score": self.scorer(case, prediction)})
        score = sum(p["score"] for p in pages) / len(pages)
        return {"benchmark": self.path, "task": self.task, "cases": pages,
                "score": score, "loss": 1.0 - score}

    def as_loss(self) -> Callable[[System], float]:
        """The benchmark as a loss function over systems: lower is better."""
        def loss(system: System) -> float:
            return float(self.run(system)["loss"])

        return loss


def load(bench_dir: str | Path) -> Exam:
    """Read a benchmark directory — benchmark.json is the whole exam, pure data."""
    data = json.loads((Path(bench_dir) / "benchmark.json").read_text())
    return Exam(data["task"], data["cases"], SCORINGS[data["scoring"]["kind"]], data["path"])


def locate(bench_root: str | Path, path: str) -> Exam:
    """Find a benchmark by ontology path /<task?>/<domain?>/<language?>."""
    for f in sorted(Path(bench_root).rglob("benchmark.json")):
        if json.loads(f.read_text())["path"] == path:
            return load(f.parent)
    raise LookupError(f"no benchmark with ontology path {path!r}")