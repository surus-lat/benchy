"""BENCHMARK — task + data + scoring. The exam. The system is the argument.

DISSOLUTION ATTEMPT (cycle 15): Benchmark the class replaced by free
functions over a plain tuple bench = (task, scoring, cases).

    run(bench, system) -> artifact
    as_loss(bench)     -> (System) -> float
"""

from .scoring import score


def run(bench, system) -> dict:
    """the system takes the exam. Returns the graded artifact."""
    task, scoring, exam = bench
    per_case = []
    for inp, expected in exam:
        pred = system(inp)
        per_case.append({
            "input": inp,
            "expected": expected,
            "prediction": pred,
            "score": score(scoring, pred, expected),
        })
    exam_score = sum(c["score"] for c in per_case) / len(per_case)
    return {"score": exam_score, "cases": per_case}


def as_loss(bench):
    """(System) -> float. The benchmark AS a loss function."""
    def loss(system):
        return 1.0 - run(bench, system)["score"]
    return loss