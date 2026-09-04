"""benchy.scoring — scoring is a symbolic program, not a class-per-rubric.

VISION.md says the scoring function is how a business defines "good", so a
rubric must be composable (small pieces combine), introspectable (reading it
tells you what the benchmark means), and transformable (operators build new
scorers from old). Authoring a rubric is a one-liner; adding a *family* is
the rare contributor activity (write a factory, `register_scorer` it).

    default  = field_wise(fields=INVOICE_FIELDS, per_field=exact_match())
    strict   = binary(default)
    weighted = field_wise_weighted(fields=INVOICE_FIELDS, per_field=exact_match(),
                                   weights={"vendor": 1, "total": 3, "due_date": 2})

Every factory in this package returns an object satisfying `benchy.core
.Scorer` (a `runtime_checkable` Protocol — `isinstance(exact_match(),
benchy.core.Scorer)` is True):

  - `evaluate(prediction, expected, sample) -> Score` — rich; `Score.value`
    is a float in `[0.0, 1.0]`, `Score.breakdown` is the why (shape
    documented per-scorer-family below).
  - `fitness(prediction, expected, sample) -> float` — the scalar an
    optimizer consumes. **Always in `[0.0, 1.0]`, always higher-is-better** —
    error-family scorers (`wer`, `cer`, `mse`) invert internally so this
    holds even for metrics that are "lower is better" in the literature; see
    `benchy.scoring.primitives`'s module docstring for the exact formulas.
  - `aggregate(scores) -> Mapping[str, Any]` — over a run; always includes a
    `"fitness"` key (the run-level scalar an engine/optimizer reads with no
    rescaling), plus whatever else is informative for that family (e.g.
    `wer`'s aggregate also reports the raw mean `"wer"`).
  - `__repr__()` that **round-trips**: `parse_scorer(repr(s))` reconstructs
    an equal scorer. A scorer's repr is a Python call expression naming a
    registered factory with literal/nested-scorer keyword arguments —
    `binary(field_wise(fields=('vendor', 'total'), per_field=exact_match()))`
    is valid input to `parse_scorer`. See `benchy.scoring.registry` for the
    grammar and its (deliberate) safety scope.

Package layout:

    base.py         BaseScorer (the ABC every scorer extends) + clamp01/_mean.
    primitives.py   atomic scorers: exact_match, contains, regex_match,
                    f1_token, numeric_close, wer, cer, mse, pearson, iou,
                    levenshtein_ratio, multiple_choice_accuracy.
    structural.py   field_wise, field_wise_weighted, list_wise — scorers that
                    combine a per-field/per-item Scorer over a structured
                    prediction/expected pair.
    transforms.py   binary, threshold, restrict, mean, weighted_sum, invert,
                    clamp — Scorer -> Scorer (or Scorers -> Scorer) operators.
    registry.py     register_scorer / registry / parse_scorer.
    _text.py, _choice.py   private helpers (normalization, choice parsing).

Two design decisions worth knowing before composing scorers (documented in
full where they're implemented):

  - **Null handling** (`primitives.py`, `structural.py`): `exact_match` and
    `numeric_close` treat `(None, None)` as an exact match, because in
    structured extraction `null` is frequently the *correct* answer, not a
    missing value. `field_wise` distinguishes "field absent from the
    prediction" (tracked in `breakdown["missing"]`) from "field present with
    value `None`" (scored normally) on exactly that basis.
  - **`field_wise`'s `Score.breakdown` shape** (relevant to any engine/report
    code rendering it): `{"fields": {"<path>": {"value", "weight",
    "breakdown"}, ...}, "missing": ["<path>", ...]}`. `list_wise`'s shape is
    `{"items": [{"value", "breakdown"}, ...], "n_predicted", "n_expected"}`.
"""

from __future__ import annotations

from .base import BaseScorer
from .primitives import (
    cer,
    contains,
    exact_match,
    f1_token,
    iou,
    levenshtein_ratio,
    mse,
    multiple_choice_accuracy,
    numeric_close,
    pearson,
    regex_match,
    wer,
)
from .registry import parse_scorer, register_scorer, registry
from .structural import field_wise, field_wise_weighted, list_wise
from .transforms import binary, clamp, invert, mean, restrict, threshold, weighted_sum

__all__ = [
    "BaseScorer",
    # atomic primitives
    "exact_match", "contains", "regex_match", "f1_token", "numeric_close",
    "wer", "cer", "multiple_choice_accuracy", "pearson", "mse", "iou",
    "levenshtein_ratio",
    # structural
    "field_wise", "field_wise_weighted", "list_wise",
    # transforms
    "binary", "threshold", "restrict", "mean", "weighted_sum", "invert", "clamp",
    # symbolic round-trip + extension
    "parse_scorer", "register_scorer", "registry",
]
