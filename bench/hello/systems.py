"""Exam-takers for /sentiment — NOT part of the benchmark: a system is any
invoked program (model, node, workflow, agent); these are offline stubs.
The cloud exam-taker (steering addendum) joins here as a spec + compiler."""
POSITIVE = ("great", "excelente", "loved")


class Good:
    """Keyword heuristic: great|excelente|loved -> pos, else neg."""
    def invoke(self, x):
        return "pos" if any(k in x for k in POSITIVE) else "neg"


class Dumb:
    """Always pos — proves the scoring discriminates (0.5)."""
    def invoke(self, x):
        return "pos"


good, dumb = Good(), Dumb()