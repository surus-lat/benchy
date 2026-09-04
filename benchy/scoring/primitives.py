"""Atomic scorer primitives.

Each factory here returns a frozen, small dataclass satisfying
`benchy.core.Scorer`. Ported from the old tree's `src/tasks/common/metrics.py`
and `image_metrics.py`, reimplemented against `BaseScorer` and the repr
round-trip contract. Heavy third-party imports (`jiwer`, `Levenshtein`) are
deferred into the method that needs them, so `import benchy.scoring` stays
instant.

Null-handling convention (this matters — read it before adding a primitive):
`exact_match` and `numeric_close` treat `(None, None)` as an **exact match**
(score 1.0), because in structured extraction `null` is frequently the
*correct* answer ("the caller never stated a business name"), not a missing
value. Any other primitive here treats `None` as invalid/unusable input
(score 0.0) — text-similarity, transcription, and choice metrics have no
comparable "the correct answer is documented as absent" case. `field_wise`
(in `benchy.scoring.structural`) relies on the exact_match/numeric_close
convention to grade nullable structured fields correctly; see its docstring
for how it distinguishes "field absent from the prediction entirely" from
"field present with value `None`".

Error-family metrics (`wer`, `cer`, `mse`) invert so higher is always better:
`wer`/`cer` compute `value = clamp(1 - raw_rate, 0, 1)` (the raw, uninverted
rate lives in `breakdown["wer"]` / `breakdown["cer"]` — jiwer's WER can
exceed 1.0 when there are more insertions than reference words, hence the
clamp); `mse` computes `value = 1 / (1 + raw_mse)`, since raw MSE is
unbounded above and `1 - mse` would go negative. Both formulas map "no
error" to fitness 1.0 and approach 0.0 as error grows, monotonically.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from benchy.core import Sample, Score

from ._choice import parse_choice_prediction
from ._text import normalize_text, tokenize
from .base import BaseScorer, clamp01, _mean
from .registry import register_scorer

__all__ = [
    "exact_match", "contains", "regex_match", "f1_token", "numeric_close",
    "wer", "cer", "multiple_choice_accuracy", "pearson", "mse", "iou",
    "levenshtein_ratio",
]


# --------------------------------------------------------------------------
# exact_match
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _ExactMatch(BaseScorer):
    """Literal equality after optional case/whitespace normalization.

    Coercion policy: both sides are stringified with `str()` before
    comparison, so `48 == "48"` and `True == "true"` compare equal — there is
    no numeric tolerance here (use `numeric_close` for that). `(None, None)`
    is an exact match (see module docstring); exactly one side `None` is not.
    """

    case_insensitive: bool = True
    strip: bool = True
    name: str = "exact_match"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        if prediction is None and expected is None:
            return Score(value=1.0, breakdown={"prediction": None, "expected": None, "match": True}, scorer=self.name)
        if prediction is None or expected is None:
            return Score(value=0.0, breakdown={"prediction": prediction, "expected": expected, "match": False}, scorer=self.name)

        pred_text = normalize_text(str(prediction), case_insensitive=self.case_insensitive, strip=self.strip)
        exp_text = normalize_text(str(expected), case_insensitive=self.case_insensitive, strip=self.strip)
        match = pred_text == exp_text
        return Score(
            value=1.0 if match else 0.0,
            breakdown={"prediction": pred_text, "expected": exp_text, "match": match},
            scorer=self.name,
        )

    def __repr__(self) -> str:
        return f"exact_match(case_insensitive={self.case_insensitive!r}, strip={self.strip!r})"


def exact_match(case_insensitive: bool = True, strip: bool = True) -> _ExactMatch:
    return _ExactMatch(case_insensitive=case_insensitive, strip=strip)


register_scorer("exact_match", exact_match)


# --------------------------------------------------------------------------
# contains
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _Contains(BaseScorer):
    """Does `str(expected)` appear as a substring of `str(prediction)`?"""

    case_insensitive: bool = True
    name: str = "contains"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        if prediction is None or expected is None:
            return Score(value=0.0, breakdown={"contains": False}, scorer=self.name)
        pred_text, exp_text = str(prediction), str(expected)
        if self.case_insensitive:
            pred_text, exp_text = pred_text.lower(), exp_text.lower()
        found = exp_text in pred_text
        return Score(value=1.0 if found else 0.0, breakdown={"contains": found}, scorer=self.name)

    def __repr__(self) -> str:
        return f"contains(case_insensitive={self.case_insensitive!r})"


def contains(case_insensitive: bool = True) -> _Contains:
    return _Contains(case_insensitive=case_insensitive)


register_scorer("contains", contains)


# --------------------------------------------------------------------------
# regex_match
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _RegexMatch(BaseScorer):
    """Does `pattern` (or, if `pattern is None`, `str(expected)` used as the
    pattern) `re.search` against `str(prediction)`?"""

    pattern: str | None
    flags: int = 0
    name: str = "regex_match"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        if prediction is None:
            return Score(value=0.0, breakdown={"matched": False}, scorer=self.name)
        pattern = self.pattern if self.pattern is not None else (str(expected) if expected is not None else None)
        if pattern is None:
            return Score(value=0.0, breakdown={"matched": False, "error": "no pattern"}, scorer=self.name)
        try:
            matched = re.search(pattern, str(prediction), self.flags) is not None
        except re.error as exc:
            return Score(value=0.0, breakdown={"matched": False, "error": str(exc)}, scorer=self.name)
        return Score(value=1.0 if matched else 0.0, breakdown={"matched": matched, "pattern": pattern}, scorer=self.name)

    def __repr__(self) -> str:
        return f"regex_match(pattern={self.pattern!r}, flags={self.flags!r})"


def regex_match(pattern: str | None = None, flags: int = 0) -> _RegexMatch:
    return _RegexMatch(pattern=pattern, flags=flags)


register_scorer("regex_match", regex_match)


# --------------------------------------------------------------------------
# f1_token
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _F1Token(BaseScorer):
    """Whitespace-token-level F1. If `expected` is a list of references, the
    max F1 over all references is used (multi-reference grading)."""

    name: str = "f1_token"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        pred_tokens = tokenize(prediction)
        if isinstance(expected, list):
            candidates = [self._prf1(pred_tokens, tokenize(ref)) for ref in expected] or [(0.0, 0.0, 0.0)]
            precision, recall, f1 = max(candidates, key=lambda prf: prf[2])
        else:
            precision, recall, f1 = self._prf1(pred_tokens, tokenize(expected))
        return Score(value=clamp01(f1), breakdown={"precision": precision, "recall": recall, "f1": f1}, scorer=self.name)

    @staticmethod
    def _prf1(pred_tokens: list[str], exp_tokens: list[str]) -> tuple[float, float, float]:
        if not pred_tokens or not exp_tokens:
            return (0.0, 0.0, 0.0)
        remaining = {}
        for tok in exp_tokens:
            remaining[tok] = remaining.get(tok, 0) + 1
        overlap = 0
        for tok in pred_tokens:
            if remaining.get(tok, 0) > 0:
                overlap += 1
                remaining[tok] -= 1
        precision = overlap / len(pred_tokens)
        recall = overlap / len(exp_tokens)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        return (precision, recall, f1)

    def __repr__(self) -> str:
        return "f1_token()"


def f1_token() -> _F1Token:
    return _F1Token()


register_scorer("f1_token", f1_token)


# --------------------------------------------------------------------------
# numeric_close
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _NumericClose(BaseScorer):
    """Numeric closeness with tolerance. `1.0` within `abs_tol + rel_tol *
    |expected|`; otherwise a graded score `max(0, 1 - relative_error)` so a
    near-miss still earns partial credit. `(None, None)` is an exact match
    (see module docstring)."""

    rel_tol: float = 1e-3
    abs_tol: float = 1e-6
    name: str = "numeric_close"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        if prediction is None and expected is None:
            return Score(value=1.0, breakdown={"diff": 0.0, "within_tolerance": True}, scorer=self.name)
        if prediction is None or expected is None:
            return Score(value=0.0, breakdown={"diff": None, "within_tolerance": False}, scorer=self.name)
        try:
            pred_val = float(prediction)
            exp_val = float(expected)
        except (TypeError, ValueError):
            return Score(value=0.0, breakdown={"error": "non-numeric input"}, scorer=self.name)

        diff = abs(pred_val - exp_val)
        tol = self.abs_tol + self.rel_tol * abs(exp_val)
        within = diff <= tol
        if within:
            value = 1.0
        else:
            denom = abs(exp_val) if exp_val != 0 else max(abs(pred_val), 1.0)
            relative_error = diff / denom if denom else diff
            value = clamp01(1.0 - relative_error)
        return Score(value=value, breakdown={"diff": diff, "within_tolerance": within}, scorer=self.name)

    def __repr__(self) -> str:
        return f"numeric_close(rel_tol={self.rel_tol!r}, abs_tol={self.abs_tol!r})"


def numeric_close(rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> _NumericClose:
    return _NumericClose(rel_tol=rel_tol, abs_tol=abs_tol)


register_scorer("numeric_close", numeric_close)


# --------------------------------------------------------------------------
# wer / cer
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _WordErrorRate(BaseScorer):
    """Word Error Rate, inverted (see module docstring)."""

    name: str = "wer"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        if not prediction or not expected:
            raw = 1.0
        else:
            try:
                import jiwer

                raw = float(jiwer.wer(str(expected), str(prediction)))
            except Exception:
                raw = 1.0
        return Score(value=clamp01(1.0 - raw), breakdown={"wer": raw}, scorer=self.name)

    def aggregate(self, scores):
        fitness = _mean([s.value for s in scores])
        raw_vals = [s.breakdown.get("wer", 1.0 - s.value) for s in scores]
        return {"fitness": fitness, "wer": _mean(raw_vals), "n": len(scores)}

    def __repr__(self) -> str:
        return "wer()"


def wer() -> _WordErrorRate:
    return _WordErrorRate()


register_scorer("wer", wer)


@dataclass(frozen=True, repr=False)
class _CharErrorRate(BaseScorer):
    """Character Error Rate, inverted (see module docstring)."""

    name: str = "cer"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        if not prediction or not expected:
            raw = 1.0
        else:
            try:
                import jiwer

                raw = float(jiwer.cer(str(expected), str(prediction)))
            except Exception:
                raw = 1.0
        return Score(value=clamp01(1.0 - raw), breakdown={"cer": raw}, scorer=self.name)

    def aggregate(self, scores):
        fitness = _mean([s.value for s in scores])
        raw_vals = [s.breakdown.get("cer", 1.0 - s.value) for s in scores]
        return {"fitness": fitness, "cer": _mean(raw_vals), "n": len(scores)}

    def __repr__(self) -> str:
        return "cer()"


def cer() -> _CharErrorRate:
    return _CharErrorRate()


register_scorer("cer", cer)


# --------------------------------------------------------------------------
# mse
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _MeanSquaredError(BaseScorer):
    """Squared error, inverted via `1 / (1 + mse)` (see module docstring).
    `(None, None)` is a perfect fitness, matching `numeric_close`."""

    name: str = "mse"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        if prediction is None and expected is None:
            return Score(value=1.0, breakdown={"mse": 0.0}, scorer=self.name)
        try:
            raw = (float(prediction) - float(expected)) ** 2
        except (TypeError, ValueError):
            return Score(value=0.0, breakdown={"mse": None, "error": "non-numeric input"}, scorer=self.name)
        return Score(value=clamp01(1.0 / (1.0 + raw)), breakdown={"mse": raw}, scorer=self.name)

    def aggregate(self, scores):
        fitness = _mean([s.value for s in scores])
        raw_vals = [s.breakdown.get("mse") for s in scores if s.breakdown.get("mse") is not None]
        return {"fitness": fitness, "mse": _mean(raw_vals) if raw_vals else 0.0, "n": len(scores)}

    def __repr__(self) -> str:
        return "mse()"


def mse() -> _MeanSquaredError:
    return _MeanSquaredError()


register_scorer("mse", mse)


# --------------------------------------------------------------------------
# pearson
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _Pearson(BaseScorer):
    """Pearson correlation is a run-level statistic, not a per-sample one.

    `evaluate` returns a *validity* indicator (1.0 if the pair is usable
    numeric data, 0.0 otherwise) and stashes the raw pair in `breakdown` for
    `aggregate` to compute the real correlation coefficient across the run.
    Judge pearson quality via `aggregate(...)["pearson"]` /
    `["fitness"]` (which maps `r` from `[-1, 1]` into `[0, 1]` via
    `(r + 1) / 2`), not via per-sample `fitness`.
    """

    name: str = "pearson"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        try:
            pred_val = float(prediction)
            exp_val = float(expected)
            if not (math.isfinite(pred_val) and math.isfinite(exp_val)):
                raise ValueError
        except (TypeError, ValueError):
            return Score(value=0.0, breakdown={"valid": False}, scorer=self.name)
        return Score(
            value=1.0,
            breakdown={"prediction": pred_val, "expected": exp_val, "valid": True},
            scorer=self.name,
        )

    def aggregate(self, scores):
        pairs = [
            (s.breakdown["prediction"], s.breakdown["expected"])
            for s in scores
            if s.breakdown.get("valid")
        ]
        if len(pairs) < 2:
            return {"fitness": 0.5, "pearson": 0.0, "n": len(scores)}

        preds = [p for p, _ in pairs]
        exps = [e for _, e in pairs]
        mean_p, mean_e = _mean(preds), _mean(exps)
        numerator = sum((p - mean_p) * (e - mean_e) for p, e in pairs)
        denom_p = sum((p - mean_p) ** 2 for p in preds)
        denom_e = sum((e - mean_e) ** 2 for e in exps)
        denom = math.sqrt(denom_p * denom_e)
        r = 0.0 if denom == 0 else max(-1.0, min(1.0, numerator / denom))
        return {"fitness": clamp01((r + 1.0) / 2.0), "pearson": r, "n": len(scores)}

    def __repr__(self) -> str:
        return "pearson()"


def pearson() -> _Pearson:
    return _Pearson()


register_scorer("pearson", pearson)


# --------------------------------------------------------------------------
# iou
# --------------------------------------------------------------------------

def _is_bbox(value) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


@dataclass(frozen=True, repr=False)
class _IoU(BaseScorer):
    """Generic intersection-over-union. A length-4 numeric sequence on both
    sides is treated as a bounding box `(x1, y1, x2, y2)`; anything else
    iterable is treated as a set (pixel coordinates, label ids, tokens) and
    scored via Jaccard similarity, which is literally IoU for masks
    represented as coordinate sets. No imaging dependency required — decode
    pixels/masks to sets or boxes before calling this."""

    name: str = "iou"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        if prediction is None or expected is None:
            return Score(value=0.0, breakdown={"error": "missing value"}, scorer=self.name)

        if _is_bbox(prediction) and _is_bbox(expected):
            value = _bbox_iou(prediction, expected)
            return Score(value=clamp01(value), breakdown={"kind": "bbox"}, scorer=self.name)

        pred_set, exp_set = set(prediction), set(expected)
        if not pred_set and not exp_set:
            value = 1.0
        else:
            union = pred_set | exp_set
            value = len(pred_set & exp_set) / len(union) if union else 1.0
        return Score(
            value=clamp01(value),
            breakdown={"kind": "set", "intersection": len(pred_set & exp_set), "union": len(pred_set | exp_set)},
            scorer=self.name,
        )

    def __repr__(self) -> str:
        return "iou()"


def iou() -> _IoU:
    return _IoU()


register_scorer("iou", iou)


# --------------------------------------------------------------------------
# levenshtein_ratio
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _LevenshteinRatio(BaseScorer):
    """Normalized Levenshtein similarity in `[0, 1]` (already higher-is-better,
    no inversion needed)."""

    name: str = "levenshtein_ratio"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        if prediction is None or expected is None:
            return Score(value=0.0, breakdown={"ratio": 0.0}, scorer=self.name)
        import Levenshtein

        ratio = float(Levenshtein.ratio(str(prediction), str(expected)))
        return Score(value=clamp01(ratio), breakdown={"ratio": ratio}, scorer=self.name)

    def __repr__(self) -> str:
        return "levenshtein_ratio()"


def levenshtein_ratio() -> _LevenshteinRatio:
    return _LevenshteinRatio()


register_scorer("levenshtein_ratio", levenshtein_ratio)


# --------------------------------------------------------------------------
# multiple_choice_accuracy
# --------------------------------------------------------------------------

@dataclass(frozen=True, repr=False)
class _MultipleChoiceAccuracy(BaseScorer):
    """Accuracy over a multiple-choice prediction.

    Reads the option texts from `sample.meta["choices"]` (a sequence of
    strings), and optionally `sample.meta["choice_labels"]` (custom letters)
    and `sample.meta["label_to_index"]` (numeric-label remapping) — this is
    the contract a Task must populate for this scorer to work.
    `prediction` may be a 0-based index, a label letter, JSON, or freeform
    text naming a choice. `expected` is the correct 0-based index or a
    label/choice string. Predictions that can't be resolved to a choice are
    excluded from `aggregate`'s accuracy (marked `"valid": False`), not
    counted as incorrect — they measure format-following failure, not
    extraction failure.
    """

    strict: bool = True
    name: str = "multiple_choice_accuracy"

    def evaluate(self, prediction, expected, sample: Sample | None = None) -> Score:
        meta = getattr(sample, "meta", None) or {}
        choices = meta.get("choices") or []
        labels = meta.get("choice_labels")
        label_to_index = meta.get("label_to_index")

        parsed = parse_choice_prediction(
            prediction, choices, labels=labels, label_to_index=label_to_index, strict=self.strict
        )
        if parsed is None:
            return Score(value=0.0, breakdown={"valid": False, "parsed": None}, scorer=self.name)

        expected_index = self._resolve_expected_index(expected, choices, labels, label_to_index)
        correct = parsed == expected_index
        return Score(
            value=1.0 if correct else 0.0,
            breakdown={"valid": True, "parsed": parsed, "expected_index": expected_index, "correct": correct},
            scorer=self.name,
        )

    @staticmethod
    def _resolve_expected_index(expected, choices, labels, label_to_index):
        if isinstance(expected, bool):
            return int(expected)
        if isinstance(expected, int):
            return expected
        if isinstance(expected, str):
            parsed = parse_choice_prediction(expected, choices, labels=labels, label_to_index=label_to_index, strict=False)
            return parsed if parsed is not None else expected
        return expected

    def aggregate(self, scores):
        valid = [s for s in scores if s.breakdown.get("valid")]
        fitness = _mean([s.value for s in valid]) if valid else 0.0
        return {"fitness": fitness, "n": len(scores), "n_valid": len(valid)}

    def __repr__(self) -> str:
        return f"multiple_choice_accuracy(strict={self.strict!r})"


def multiple_choice_accuracy(strict: bool = True) -> _MultipleChoiceAccuracy:
    return _MultipleChoiceAccuracy(strict=strict)


register_scorer("multiple_choice_accuracy", multiple_choice_accuracy)
