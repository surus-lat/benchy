"""nb.exam — the exam engine: benchmark = task + cases + scoring; the system
is the argument.  run(system) grades a taker;  as_loss() exports the loss."""
import json
from pathlib import Path


class Exam:
    """One benchmark.  run(system) grades a taker;  as_loss() exports the loss."""
    # survived dissolution (cycle 5): the class is the concept-compressor —
    # it carries the vision invariant's own syntax (GOLEM law 6:
    # benchmark.run(system) / benchmark.as_loss()) and hides the exam's
    # internal shape; a 3-tuple leaked that shape to every caller (cli had
    # to destructure AND re-wrap, plus a run/run collision).
    # cycle 7: the exam is PURE — cases + scoring.  path/dir attributes
    # died: addresses are the caller's business (the CLI derives them from
    # the ontology path it already holds); the artifact's identity fields
    # died with them (the filename IS the identity, cycle 3's finding).

    def __init__(self, cases, scorer):
        self.cases, self.scorer = cases, scorer

    def run(self, system) -> dict:
        """The system takes the exam; returns the graded artifact (JSON-ready)."""
        pages = []
        for case in self.cases:
            prediction = system.invoke(case["input"])
            pages.append({**case, "prediction": prediction,
                          "score": self.scorer(case, prediction)})
        return {"cases": pages,
                "score": sum(p["score"] for p in pages) / len(pages)}

    def as_loss(self):
        """The benchmark as a loss function over systems: lower is better."""
        def loss(system):
            return 1.0 - self.run(system)["score"]
        return loss


def locate(bench_root, path):
    """Resolve an ontology path /<task?>/<domain?>/<language?> to its exam.
    benchmark.json is the whole benchmark: {task, cases} — data only."""
    # flat lookup, no walk (cycle 6): the ontology path IS a directory path
    # under bench/ — the vision's /<task?>/<domain?>/<language?> is literally
    # the filesystem.  The walk existed only to reconcile the data's `path`
    # field (a second address, cycle-3's crime repeated) with the directory;
    # honest-by-construction beats a registry that re-derives the address,
    # and a sibling benchmark can no longer break an unrelated locate.
    f = Path(bench_root) / path.lstrip("/") / "benchmark.json"
    if not f.exists():
        raise LookupError(f"no benchmark with ontology path {path!r} under {bench_root}")
    data = json.loads(f.read_text())
    # the exam is EXACTLY {task, cases} (cycle 9: was an unenforced claim —
    # a task-less exam loaded fine while the error message promised the
    # format; the exact-set check names the actual keys, so a missing
    # pillar and a junk key are both diagnosable from one honest message).
    # task content is the TAKER's business (the cloud compiler reads it to
    # build prompts) — presence is load-time honesty, semantics stay free.
    if set(data) != {"task", "cases"}:
        raise ValueError(f"{f}: an exam is exactly {{task, cases}}; got {sorted(data)}")
    if not data["cases"]:
        raise ValueError(f"{f}: no cases — nothing to grade; add cases to benchmark.json")
    # exact-match scoring: 1 point per exact match (the hello bar); the
    # task's output enum is what the cloud compiler reads — the engine
    # never looks (cycle 1: the engine's task plumbing was a lens).
    exact = lambda case, prediction: float(prediction == case["expected"])
    return Exam(data["cases"], exact)