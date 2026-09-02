"""DATA — the exam. n cases. Cases are data, never code.

A case is (input, expected) — the system takes the exam by invoking on
each input; grading compares each prediction to expected.
Cases may live in JSON (bench/hello/cases.json) — the engine never
requires Python to define an exam.
"""


from .benchmark import Case


class Exam:
    """the exam: a list of (input, expected) cases."""

    def __init__(self, cases):
        if not cases:
            raise ValueError("an exam needs at least one case")
        self.cases = [
            c if isinstance(c, Case) else Case(c["input"], c["expected"])
            for c in cases
        ]

    def __len__(self):
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)