"""SCORING — the grading function. What good means. And the loss.

Scoring is declared as DATA: scoring.json = {"mode": ..., "weights": ...}.
`score` maps one prediction to 0..1; loss is 1 - exam score.
"""


def score(spec, pred, expected) -> float:
    """grade one prediction 0..1. spec = the scoring.json dict (data).

    exact:    pred == expected -> 1 else 0
    partial:  per-field match fraction of an expected dict
    weighted: per-field match, each field's contribution fixed by
              spec["weights"] (business importance). Unmentioned fields
              count 0; a weightless weighted spec degrades to partial.
    """
    mode = spec.get("mode", "exact")
    if mode == "exact" or not isinstance(expected, dict):
        return 1.0 if pred == expected else 0.0
    got = pred.get if isinstance(pred, dict) else (lambda _: None)
    fields = list(expected)
    if mode == "weighted":
        weights = spec.get("weights") or {}
        total = sum(weights.get(f, 0.0) for f in fields)
        if total <= 0:  # no weights given: weighted degrades to partial
            return sum(1 for f in fields if got(f) == expected[f]) / len(fields)
        return sum(weights.get(f, 0.0) for f in fields
                   if got(f) == expected[f]) / total
    # partial
    return sum(1 for f in fields if got(f) == expected[f]) / len(fields)