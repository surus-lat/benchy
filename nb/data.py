"""DATA — the exam. n cases. Cases are data, never code.

A case is (input, expected) — a plain tuple. The system takes the exam
by being invoked on each input; grading compares each prediction to
expected. Cases may live in JSON (bench/hello/cases.json) — the engine
never requires Python to define an exam.
"""


class Exam:
    """the exam: a list of (input, expected) tuples."""

    def __init__(self, cases):
        if not cases:
            raise ValueError("an exam needs at least one case")
        self.cases = list(cases)

    def __iter__(self):
        return iter(self.cases)