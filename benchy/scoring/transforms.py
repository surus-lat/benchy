"""Transforms: scorer -> scorer (or scorers -> scorer) operators.

"Operators build new scorers from old" — this module is where that half of
the vision lives. Every factory here takes at least one `Scorer` and returns
a new one; composing them is how `binary(field_wise_weighted(...))`-style
rubrics get built without a new class per rubric.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from benchy.core import Sample, Score, Scorer

from .base import BaseScorer, _mean, clamp01
from .registry import register_scorer
from .structural import _PATH_TOKEN_RE, _resolve_path

__all__ = ["binary", "threshold", "restrict", "mean", "weighted_sum", "invert", "clamp"]


# --------------------------------------------------------------------------
# binary
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _Binary(BaseScorer):
    """Collapse `inner`'s graded score to {0.0, 1.0} at `cutoff`."""

    inner: Scorer
    cutoff: float = 0.5

    name = "binary"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        inner_score = self.inner.evaluate(prediction, expected, sample)
        value = 1.0 if inner_score.value >= self.cutoff else 0.0
        return Score(
            value=value,
            breakdown={"inner": inner_score.value, "inner_breakdown": inner_score.breakdown, "cutoff": self.cutoff},
            scorer=self.name,
        )

    def __repr__(self) -> str:
        return f"binary({self.inner!r}, cutoff={self.cutoff!r})"


def binary(inner: Scorer, cutoff: float = 0.5) -> _Binary:
    return _Binary(inner=inner, cutoff=cutoff)


register_scorer("binary", binary)


# --------------------------------------------------------------------------
# threshold
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _Threshold(BaseScorer):
    """Below `cutoff`, floor to `floor`; at or above it, keep the graded
    value. Unlike `binary`, a passing score is not rounded up to 1.0 — this
    is "you must clear a quality bar to count at all," not "pass/fail."""

    inner: Scorer
    cutoff: float
    floor: float = 0.0

    name = "threshold"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        inner_score = self.inner.evaluate(prediction, expected, sample)
        value = inner_score.value if inner_score.value >= self.cutoff else self.floor
        return Score(
            value=value,
            breakdown={
                "inner": inner_score.value,
                "inner_breakdown": inner_score.breakdown,
                "cutoff": self.cutoff,
                "floor": self.floor,
            },
            scorer=self.name,
        )

    def __repr__(self) -> str:
        return f"threshold({self.inner!r}, cutoff={self.cutoff!r}, floor={self.floor!r})"


def threshold(inner: Scorer, cutoff: float, floor: float = 0.0) -> _Threshold:
    return _Threshold(inner=inner, cutoff=cutoff, floor=floor)


register_scorer("threshold", threshold)


# --------------------------------------------------------------------------
# restrict
# --------------------------------------------------------------------------

def _set_path(container: dict, path: str, value: Any) -> None:
    tokens = [key for key, _idx in _PATH_TOKEN_RE.findall(path) if key]
    if not tokens:
        return
    current = container
    for token in tokens[:-1]:
        current = current.setdefault(token, {})
    current[tokens[-1]] = value


def _project(obj: Mapping, fields: Sequence[str]) -> dict:
    result: dict = {}
    for field in fields:
        found, value = _resolve_path(obj, field)
        if found:
            _set_path(result, field, value)
    return result


@dataclass(frozen=True, repr=False)
class _Restrict(BaseScorer):
    """Project `prediction` and `expected` down to `fields` (dotted paths;
    array-indexed paths are not reconstructed) before delegating to `inner`.
    Non-mapping inputs pass through unchanged — restrict is then a no-op.
    Lets a big rubric be re-graded on a slice of its fields without
    rebuilding it (e.g. "just the header fields" from a full-invoice
    `field_wise`)."""

    inner: Scorer
    fields: tuple[str, ...]

    name = "restrict"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        proj_pred = _project(prediction, self.fields) if isinstance(prediction, Mapping) else prediction
        proj_exp = _project(expected, self.fields) if isinstance(expected, Mapping) else expected
        return self.inner.evaluate(proj_pred, proj_exp, sample)

    def __repr__(self) -> str:
        return f"restrict({self.inner!r}, fields={self.fields!r})"


def restrict(inner: Scorer, fields: Sequence[str]) -> _Restrict:
    return _Restrict(inner=inner, fields=tuple(fields))


register_scorer("restrict", restrict)


# --------------------------------------------------------------------------
# mean
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _Mean(BaseScorer):
    """Unweighted average of several independent scorers on the same pair."""

    scorers: tuple[Scorer, ...]

    name = "mean"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        sub_scores = [s.evaluate(prediction, expected, sample) for s in self.scorers]
        value = _mean([sc.value for sc in sub_scores]) if sub_scores else 0.0
        breakdown = {
            "components": [
                {"scorer": repr(s), "value": sc.value, "breakdown": sc.breakdown}
                for s, sc in zip(self.scorers, sub_scores)
            ]
        }
        return Score(value=clamp01(value), breakdown=breakdown, scorer=self.name)

    def __repr__(self) -> str:
        return f"mean({', '.join(repr(s) for s in self.scorers)})"


def mean(*scorers: Scorer) -> _Mean:
    return _Mean(scorers=tuple(scorers))


register_scorer("mean", mean)


# --------------------------------------------------------------------------
# weighted_sum
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _WeightedSum(BaseScorer):
    """A named, weighted combination of independent scorers. Despite the
    name this normalizes by total weight (a weighted mean) so the result
    always stays in [0, 1] regardless of what the weights sum to — named
    `weighted_sum` to parallel `field_wise_weighted`'s `weights=` keyword,
    and because each component contributes `weight * value` to the total
    before normalization."""

    scorers: Mapping[str, Scorer]
    weights: Mapping[str, float]

    name = "weighted_sum"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        components: dict[str, Any] = {}
        weighted_total = 0.0
        total_weight = 0.0
        for key, scorer in self.scorers.items():
            sub_score = scorer.evaluate(prediction, expected, sample)
            weight = float(self.weights.get(key, 1.0))
            components[key] = {"value": sub_score.value, "weight": weight, "breakdown": sub_score.breakdown}
            weighted_total += weight * sub_score.value
            total_weight += weight
        value = weighted_total / total_weight if total_weight > 0 else 0.0
        return Score(value=clamp01(value), breakdown={"components": components}, scorer=self.name)

    def __repr__(self) -> str:
        scorers_repr = "{" + ", ".join(f"{k!r}: {v!r}" for k, v in self.scorers.items()) + "}"
        return f"weighted_sum({scorers_repr}, weights={dict(self.weights)!r})"


def weighted_sum(scorers: Mapping[str, Scorer], weights: Mapping[str, float]) -> _WeightedSum:
    return _WeightedSum(scorers=dict(scorers), weights=dict(weights))


register_scorer("weighted_sum", weighted_sum)


# --------------------------------------------------------------------------
# invert
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _Invert(BaseScorer):
    """Flip `inner`'s value: `1 - value`, clamped to [0, 1]. For wrapping a
    hand-rolled scorer whose author built it "lower is better" and forgot to
    invert (the built-in error-family primitives already invert themselves —
    see `benchy.scoring.primitives`)."""

    inner: Scorer

    name = "invert"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        inner_score = self.inner.evaluate(prediction, expected, sample)
        value = clamp01(1.0 - inner_score.value)
        return Score(
            value=value,
            breakdown={"inner": inner_score.value, "inner_breakdown": inner_score.breakdown},
            scorer=self.name,
        )

    def __repr__(self) -> str:
        return f"invert({self.inner!r})"


def invert(inner: Scorer) -> _Invert:
    return _Invert(inner=inner)


register_scorer("invert", invert)


# --------------------------------------------------------------------------
# clamp
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _Clamp(BaseScorer):
    """Clip `inner`'s value into `[lo, hi]`."""

    inner: Scorer
    lo: float = 0.0
    hi: float = 1.0

    name = "clamp"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        inner_score = self.inner.evaluate(prediction, expected, sample)
        value = max(self.lo, min(self.hi, inner_score.value))
        return Score(
            value=value,
            breakdown={"inner": inner_score.value, "inner_breakdown": inner_score.breakdown, "lo": self.lo, "hi": self.hi},
            scorer=self.name,
        )

    def __repr__(self) -> str:
        return f"clamp({self.inner!r}, lo={self.lo!r}, hi={self.hi!r})"


def clamp(inner: Scorer, lo: float = 0.0, hi: float = 1.0) -> _Clamp:
    return _Clamp(inner=inner, lo=lo, hi=hi)


register_scorer("clamp", clamp)
