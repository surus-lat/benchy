"""nb.exam — the exam engine: benchmark = task + cases + scoring; the system
is the argument.  run(system) grades a taker;  as_loss() exports the loss."""
import json
from pathlib import Path


class Exam:
    """One benchmark.  run(system) grades a taker;  as_loss() exports the loss."""

    def __init__(self, cases, scorer, path="", dir=None):
        self.cases, self.scorer = cases, scorer
        self.path, self.dir = path, dir

    def run(self, system) -> dict:
        """The system takes the exam; returns the graded artifact (JSON-ready)."""
        pages = []
        for case in self.cases:
            prediction = system.invoke(case["input"])
            pages.append({**case, "prediction": prediction,
                          "score": self.scorer(case, prediction)})
        return {"benchmark": self.path, "cases": pages,
                "score": sum(p["score"] for p in pages) / len(pages)}

    def as_loss(self):
        """The benchmark as a loss function over systems: lower is better."""
        def loss(system):
            return 1.0 - self.run(system)["score"]
        return loss


def locate(bench_root, path):
    """Resolve an ontology path /<task?>/<domain?>/<language?> to its exam.
    benchmark.json is the whole benchmark: {path, task, cases} — data only."""
    for f in sorted(Path(bench_root).rglob("benchmark.json")):
        data = json.loads(f.read_text())
        unknown = set(data) - {"path", "task", "cases"}
        if unknown:
            raise ValueError(f"{f}: unknown keys {sorted(unknown)} — an exam is {{path, task, cases}}")
        if data["path"] == path:
            if not data["cases"]:
                raise ValueError(f"{f}: no cases — nothing to grade; "
                                 "add cases to benchmark.json")
            exact = lambda case, prediction: float(prediction == case["expected"])
            return Exam(data["cases"], exact, path, f.parent)
    raise LookupError(f"no benchmark with ontology path {path!r} under {bench_root}")