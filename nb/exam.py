"""nb.exam — the exam: benchmark = data + scoring; system is the argument.

The only behavior in the engine: a for-loop over cases, a mean, and JSON I/O.
"""
import json
import sys
from pathlib import Path


class Exam:
    """One benchmark.  run(system) grades a taker; as_loss() exports the loss."""

    def __init__(self, cases, scorer, path: str = "", dir=None):
        self.cases, self.scorer, self.path, self.dir = cases, scorer, path, dir

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

    def as_loss(self):
        """The benchmark as a loss function over systems: lower is better."""
        def loss(system) -> float:
            return float(self.run(system)["loss"])

        return loss


def locate(bench_root, path):
    """Find a benchmark by ontology path /<task?>/<domain?>/<language?> — the
    only constructor: benchmark.json is the whole exam, pure data."""
    for f in sorted(Path(bench_root).rglob("benchmark.json")):
        data = json.loads(f.read_text())
        if data["path"] == path:
            score = lambda case, prediction: float(prediction == case["expected"])
            return Exam(data["cases"], score, data["path"], f.parent)
    raise LookupError(f"no benchmark with ontology path {path!r}")


def main() -> None:
    """`python -m nb <bench_root> <ontology_path> <system>` — one exam, offline,
    end to end; writes the graded artifact next to the benchmark."""
    root, path, name = sys.argv[1], sys.argv[2], sys.argv[3]
    exam = locate(root, path)
    sys.path.insert(0, str(exam.dir))
    artifact = exam.run(getattr(__import__("stubs"), name))
    out = exam.dir / f"artifact_{name}.json"
    out.write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"{artifact['benchmark']} {name}: score={artifact['score']:.2f} "
          f"loss={artifact['loss']:.2f} -> {out}")