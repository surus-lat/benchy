"""BENCHMARK — task + data + scoring. The exam. The system is the argument.

    bench = Benchmark(task, scoring, exam)
    artifact = bench.run(system)
    loss = bench.as_loss()(system)          # (System) -> float

The graded artifact is the evidence trace of one loss evaluation: a
JSON dict with per-case scores and the aggregate.
"""


from .scoring import score


class Benchmark:
    """task + scoring + exam. The system is the argument, not a field.

    task is DATA: the {"in": ..., "out": ...} declaration from task.json.
    exam is DATA: the [(input, expected), ...] cases list. The engine
    never interprets either — the SYSTEM compiles against the task;
    scoring grades each prediction; the exam is just the evidence.
    """

    def __init__(self, task, scoring, exam):
        self.task = task
        self.scoring = scoring
        self.exam = exam

    def run(self, system) -> dict:
        """the system takes the exam. Returns the graded artifact."""
        per_case = []
        for inp, expected in self.exam:
            pred = system(inp)
            per_case.append({
                "input": inp,
                "expected": expected,
                "prediction": pred,
                "score": score(self.scoring, pred, expected),
            })
        exam_score = sum(c["score"] for c in per_case) / len(per_case)
        return {"score": exam_score, "cases": per_case}

    def as_loss(self):
        """(System) -> float. The benchmark AS a loss function."""
        def loss(system):
            return 1.0 - self.run(system)["score"]
        return loss