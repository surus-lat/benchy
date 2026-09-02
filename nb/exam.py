"""nb.exam — the exam engine: benchmark = task + cases + scoring; the system
is the argument.  run(system) grades a taker;  as_loss() exports the loss."""
import json
from pathlib import Path


class Exam:
    # survived dissolution (cycle 5): the class is the concept-compressor —
    # it carries the vision invariant's own syntax (GOLEM law 6:
    # benchmark.run(system) / benchmark.as_loss()) and hides the exam's
    # internal shape; a 3-tuple leaked that shape to every caller.
    # cycle 7: the exam is PURE — cases; addresses are the caller's.
    # cycle 11: scoring is derived engine code (IDEAS.md: from the output
    # schema), so the exam's whole state is its cases.  One field, one
    # concept.  cycle 13: grading composes the loss once (the CLI's private
    # 1-score formula died — a second address of the grading formula).

    def __init__(self, cases):
        self.cases = cases

    def run(self, system) -> dict:
        """The system takes the exam; returns the graded artifact (JSON-ready)."""
        # the SCORING pillar lives HERE (cycle 11): exact match, 1 point per
        # case, mean over the exam — the hello bar.  IDEAS.md: the scoring
        # function DERIVES from the output schema (enum -> exact match),
        # so it is engine code, not data and not a constructor seam.  The
        # scorer param died: no caller ever passed a different one, so the
        # "configurable scoring" story was speculative — when a real exam
        # needs weights, the data format grows a scoring key LOUDLY (the
        # exact-set check will demand it), never a silent Python seam.
        pages = []
        for case in self.cases:
            prediction = system.invoke(case["input"])
            pages.append({**case, "prediction": prediction,
                          "score": float(prediction == case["expected"])})
        score = sum(p["score"] for p in pages) / len(pages)
        # c13: the loss is graded HERE, once — the vision's scoring function
        # IS the loss (GOLEM law 6), so the graded artifact carries it; the
        # CLI's print and as_loss() both READ it, never re-derive it (the
        # 1-score formula had three addresses; now it has one).
        return {"cases": pages, "score": score, "loss": 1.0 - score}

    def as_loss(self):
        """The benchmark as a loss function over systems: lower is better."""
        def loss(system):
            # c13: reads the graded loss — grading owns the formula, the
            # loss-view is a projection of the evidence, not a second
            # implementation of it.
            return self.run(system)["loss"]
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
    # the exact-set check names the actual keys, so a missing pillar and a
    # junk key are both diagnosable from one honest message).  task content
    # is the TAKER's business (the cloud compiler reads it to build
    # prompts) — presence is load-time honesty, semantics stay free.
    if set(data) != {"task", "cases"}:
        raise ValueError(f"{f}: an exam is exactly {{task, cases}}; got {sorted(data)}")
    if not data["cases"]:
        raise ValueError(f"{f}: no cases — nothing to grade; add cases to benchmark.json")
    # c14: the per-case contract, enforced at LOAD — a case is at least
    # {input, expected} (extra keys — context, id — are the taker's data and
    # pass through to the artifact).  Before this, a malformed case died
    # MID-EXAM (money already spent on a cloud taker) with the cryptic
    # KeyError repr 'expected' instead of words — and the scaffold's ack
    # teaches {"input": …, "expected": …}, which the engine never checked:
    # the c9 task-lie, repeated per case.  Data errors refuse at load;
    # taker errors still crash honestly.
    for i, c in enumerate(data["cases"]):
        if "input" not in c or "expected" not in c:
            raise ValueError(f"{f}: every case needs input and expected; case {i} got {sorted(c)}")
    return Exam(data["cases"])