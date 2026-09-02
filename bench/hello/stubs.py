"""Example exam-takers for the /sentiment benchmark.

NOT part of the benchmark — benchmark.json is pure data.  A system is any
invoked program (model, node, workflow, agent); these two are offline stubs
satisfying the System contract structurally: invoke(input) -> prediction.
"""
POSITIVE = ("great", "excelente", "loved")


class Good:
    """Keyword heuristic: great|excelente|loved -> pos, else neg."""

    def invoke(self, x: str) -> str:
        return "pos" if any(k in x for k in POSITIVE) else "neg"


class Dumb:
    """Always pos — proves the scoring discriminates (0.5)."""

    def invoke(self, x: str) -> str:
        return "pos"


good, dumb = Good(), Dumb()