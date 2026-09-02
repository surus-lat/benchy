"""SCORING — the grading function. What good means. And the loss.

Scoring is declared as data. Three shapes exist, because deletion
attempts keep proving all three: exact (pos/neg, Q&A), partial (each
field of an extraction), weighted (business importance hierarchy).
`score` maps a prediction to 0..1 per case. `loss` is 1 - exam score.
"""


class Scoring:
    """exact | partial | weighted — how a case is graded.

    exact:    pred == expected -> 1 else 0
    partial:  per-field match fraction of an expected dict
    weighted: per-field match, each field's contribution fixed by
              weights (business importance). Unmentioned fields count 0.
    """

    def __init__(self, mode: str = "exact", weights: dict = None):
        self.mode = mode
        self.weights = weights or {}

    def score(self, pred, expected) -> float:
        if self.mode == "exact" or not isinstance(expected, dict):
            return 1.0 if pred == expected else 0.0
        got = pred.get if isinstance(pred, dict) else (lambda _: None)
        fields = list(expected)
        if self.mode == "weighted":
            total = sum(self.weights.get(f, 0.0) for f in fields)
            if total <= 0:  # no weights given: weighted degrades to partial
                return sum(1 for f in fields if got(f) == expected[f]) / len(fields)
            return sum(self.weights.get(f, 0.0) for f in fields
                       if got(f) == expected[f]) / total
        # partial
        return sum(1 for f in fields if got(f) == expected[f]) / len(fields)

    def as_loss(self, scored_cases) -> float:
        """the loss: 1 - mean of per-case scores. lower is better."""
        return 1.0 - sum(scored_cases) / len(scored_cases)