"""Structural scorers: field_wise, field_wise_weighted, list_wise.

`field_wise` is the highest-value salvage from the old tree (see
`src/tasks/common/utils/structured_metrics_calculator.py`, a 1300-line
god-object). The design here is deliberately narrower: `field_wise` applies
ONE `per_field` scorer uniformly across a set of dotted field paths, and all
of its partial-credit behavior comes from whichever scorer `per_field` is —
usually `exact_match()`, sometimes `numeric_close()` scoped to numeric
fields via `restrict()`. That is the composability the vision asks for:
"combine small pieces" rather than a config-driven monolith.

Three decisions, made against a real reference benchmark
(`src/tasks/structured/.data/chat_extract_data.jsonl`: nested Spanish lead
extraction with schema `{"nombre": str|null, "tiene_negocio": bool|null,
"negocio": {"descripcion_negocio": str|null, "meses_en_negocio": int|null,
"cantidad_empleados": int|null}}`) and documented here because the engine
module renders `Score.breakdown`:

1. **Dotted paths.** `fields=("nombre", "negocio.descripcion_negocio")`
   descends into nested dicts. `"items[0].codigo"`-style bracket indices
   into lists are also supported for one-off access, but `field_wise` does
   not enumerate array elements on its own — use `list_wise` for arrays of
   variable length.
2. **`fields=None`** means "every leaf path of the `expected` object for
   this sample" (recomputed per sample, since shapes can vary). A leaf is
   any value that isn't itself a nested dict — `None`, strings, numbers,
   booleans, and lists are all leaves; only dict-valued keys are descended
   into.
3. **Null handling.** A dict key that is *present* with value `None` is
   scored normally (and `exact_match`/`numeric_close` treat `(None, None)`
   as a match — see `benchy.scoring.primitives`), so a correctly-predicted
   "the caller never said their name" scores 1.0. A field that is *absent*
   from the prediction entirely is scored as if it were `None` (so it still
   counts against the score) but is additionally listed in
   `breakdown["missing"]`, so a report can distinguish "wrong value" from
   "didn't even try."

`Score.breakdown` shape for `field_wise` / `field_wise_weighted`:

    {
      "fields": {
        "<field path>": {"value": float, "weight": float, "breakdown": {...per-field breakdown...}},
        ...
      },
      "missing": ["<field path>", ...],   # fields absent from the prediction
    }

`Score.breakdown` shape for `list_wise`:

    {
      "items": [{"value": float, "breakdown": {...}}, ...],  # in pairing order
      "n_predicted": int,
      "n_expected": int,
    }
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from benchy.core import Sample, Score, Scorer

from .base import BaseScorer, _mean, clamp01
from .registry import register_scorer

__all__ = ["field_wise", "field_wise_weighted", "list_wise"]


# --------------------------------------------------------------------------
# field path resolution
# --------------------------------------------------------------------------

_PATH_TOKEN_RE = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def _resolve_path(obj: Any, path: str) -> tuple[bool, Any]:
    """Walk a dotted/bracket-indexed path. Returns `(found, value)` so a key
    present with value `None` (`found=True, value=None`) can be told apart
    from a key that doesn't exist at all (`found=False, value=None`)."""
    current = obj
    for key, idx in _PATH_TOKEN_RE.findall(path):
        if key:
            if isinstance(current, Mapping) and key in current:
                current = current[key]
            else:
                return False, None
        else:
            i = int(idx)
            if (
                isinstance(current, Sequence)
                and not isinstance(current, (str, bytes))
                and 0 <= i < len(current)
            ):
                current = current[i]
            else:
                return False, None
    return True, current


def _leaf_paths(obj: Any, prefix: str = "") -> list[str]:
    """Every dotted path to a non-dict value in a nested mapping. Lists are
    leaves (not descended into) — pair them with `list_wise` instead."""
    if isinstance(obj, Mapping):
        paths: list[str] = []
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                paths.extend(_leaf_paths(value, child_prefix))
            else:
                paths.append(child_prefix)
        return paths
    return [prefix] if prefix else []


# --------------------------------------------------------------------------
# field_wise / field_wise_weighted
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _FieldWise(BaseScorer):
    fields: tuple[str, ...] | None
    per_field: Scorer
    weights: Mapping[str, float] | None = None

    @property
    def name(self) -> str:
        return "field_wise" if self.weights is None else "field_wise_weighted"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        if self.fields is not None:
            resolved_fields: tuple[str, ...] = self.fields
        elif isinstance(expected, Mapping):
            resolved_fields = tuple(_leaf_paths(expected))
        else:
            resolved_fields = ()

        if not resolved_fields:
            return Score(value=1.0, breakdown={"fields": {}, "missing": []}, scorer=self.name)

        field_details: dict[str, Any] = {}
        missing: list[str] = []
        weighted_sum = 0.0
        total_weight = 0.0

        for field in resolved_fields:
            pred_found, pred_val = _resolve_path(prediction, field)
            exp_found, exp_val = _resolve_path(expected, field)
            pred_for_scoring = pred_val if pred_found else None
            exp_for_scoring = exp_val if exp_found else None

            sub_score = self.per_field.evaluate(pred_for_scoring, exp_for_scoring, sample)
            weight = float((self.weights or {}).get(field, 1.0))

            field_details[field] = {
                "value": sub_score.value,
                "weight": weight,
                "breakdown": sub_score.breakdown,
            }
            if not pred_found:
                missing.append(field)

            weighted_sum += weight * sub_score.value
            total_weight += weight

        value = weighted_sum / total_weight if total_weight > 0 else 1.0
        return Score(
            value=clamp01(value),
            breakdown={"fields": field_details, "missing": missing},
            scorer=self.name,
        )

    def __repr__(self) -> str:
        if self.weights is None:
            return f"field_wise(fields={self.fields!r}, per_field={self.per_field!r})"
        return (
            f"field_wise_weighted(fields={self.fields!r}, per_field={self.per_field!r}, "
            f"weights={dict(self.weights)!r})"
        )


def field_wise(fields: Sequence[str] | None, per_field: Scorer) -> _FieldWise:
    normalized = tuple(fields) if fields is not None else None
    return _FieldWise(fields=normalized, per_field=per_field, weights=None)


def field_wise_weighted(
    fields: Sequence[str] | None, per_field: Scorer, weights: Mapping[str, float]
) -> _FieldWise:
    normalized = tuple(fields) if fields is not None else None
    return _FieldWise(fields=normalized, per_field=per_field, weights=dict(weights))


register_scorer("field_wise", field_wise)
register_scorer("field_wise_weighted", field_wise_weighted)


# --------------------------------------------------------------------------
# list_wise
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _ListWise(BaseScorer):
    """Score a predicted list against an expected list, item by item.

    `order_sensitive=True` (default) pairs by index; length mismatches pad
    the shorter side with `None` (so a missing/extra item is scored, not
    ignored). `order_sensitive=False` finds the alignment that maximizes
    total item score via the Hungarian algorithm (`scipy.optimize
    .linear_sum_assignment`, lazily imported) — the right choice whenever
    the model's output order carries no meaning (unordered line items,
    detected objects, extracted entities).
    """

    item_scorer: Scorer
    order_sensitive: bool = True

    @property
    def name(self) -> str:
        return "list_wise"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        pred_list = list(prediction) if prediction is not None else []
        exp_list = list(expected) if expected is not None else []

        if not pred_list and not exp_list:
            return Score(value=1.0, breakdown={"items": [], "n_predicted": 0, "n_expected": 0}, scorer=self.name)

        pairs = self._pairs(pred_list, exp_list, sample)
        item_scores = [self.item_scorer.evaluate(p, e, sample) for p, e in pairs]
        value = _mean([sc.value for sc in item_scores])
        breakdown = {
            "items": [{"value": sc.value, "breakdown": sc.breakdown} for sc in item_scores],
            "n_predicted": len(pred_list),
            "n_expected": len(exp_list),
        }
        return Score(value=clamp01(value), breakdown=breakdown, scorer=self.name)

    def _pairs(self, pred_list: list, exp_list: list, sample: Sample | None) -> list[tuple[Any, Any]]:
        if self.order_sensitive or not pred_list or not exp_list:
            n = max(len(pred_list), len(exp_list))
            return [
                (pred_list[i] if i < len(pred_list) else None, exp_list[i] if i < len(exp_list) else None)
                for i in range(n)
            ]
        return self._aligned_pairs(pred_list, exp_list, sample)

    def _aligned_pairs(self, pred_list: list, exp_list: list, sample: Sample | None) -> list[tuple[Any, Any]]:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        n_pred, n_exp = len(pred_list), len(exp_list)
        cost = np.zeros((n_pred, n_exp))
        for i, p in enumerate(pred_list):
            for j, e in enumerate(exp_list):
                cost[i, j] = 1.0 - self.item_scorer.fitness(p, e, sample)

        row_idx, col_idx = linear_sum_assignment(cost)
        matched_pred, matched_exp = set(row_idx.tolist()), set(col_idx.tolist())
        pairs = [(pred_list[i], exp_list[j]) for i, j in zip(row_idx.tolist(), col_idx.tolist())]
        pairs.extend((pred_list[i], None) for i in range(n_pred) if i not in matched_pred)
        pairs.extend((None, exp_list[j]) for j in range(n_exp) if j not in matched_exp)
        return pairs

    def __repr__(self) -> str:
        return f"list_wise(item_scorer={self.item_scorer!r}, order_sensitive={self.order_sensitive!r})"


def list_wise(item_scorer: Scorer, order_sensitive: bool = True) -> _ListWise:
    return _ListWise(item_scorer=item_scorer, order_sensitive=order_sensitive)


register_scorer("list_wise", list_wise)
