"""Closed registry of per-field scoring metrics.

The default metric is ``exact`` — the canonical exact match of paper A.7, harsh and
impossible to game. It is also the wrong *operational* metric for some fields: a date
written "12/3/26" for "2026-03-12" is a correct extraction scored as a miss, and a
name with reordered tokens is a correct extraction scored as a miss.

This module is the single, CLOSED library a benchmark may draw from through
``scoring.field_metrics``: the benchmark author (or an optimization engine) chooses
and parametrizes, it never writes scoring code — there is no "always match" metric to
pick. Every metric maps ``(predicted, expected, params)`` to a float in [0, 1], so
partial credit is a first-class score, not a boolean dressed as one.

Text normalization (``_fold``/``_tokens``) is ported verbatim from
``program_pipeline/scoring.py``: NFKD + casefold + accent strip, alphanumeric tokens.
Tokenization parity with the optimization loop is sacred — do not "improve" it here.

``span_recall`` and ``set_f1`` are ported from datapipeline's ``derived_scoring``,
which vendors this file back (``datapipeline/scoring_metrics.py``): one registry, two
homes, byte-identical semantics.

Enum-preserving rule: a field declared ``{enum: [...]}`` may only use ENUM_SAFE
metrics. Relaxing an enum into a similarity metric would dissolve the vocabulary
contract the benchmark sealed.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any

__all__ = ["METRICS", "ENUM_SAFE", "PARAM_SPEC", "score", "catalog"]

# ---------------------------------------------------------------------------
# shared normalization helpers (ported verbatim from program_pipeline/scoring.py)


def _fold(text: Any) -> str:
    """casefold + strip accents + collapse whitespace -- the shared fold."""
    folded = unicodedata.normalize("NFKD", str(text).strip().casefold())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return " ".join(folded.split())


def _tokens(text: Any) -> frozenset:
    """Alphanumeric tokens of the folded text, order-free."""
    return frozenset(re.findall(r"[a-z0-9]+", _fold(text)))


def _token_list(text: Any) -> list:
    """Folded tokens as a list (repetitions kept) — span_recall counts them."""
    return re.findall(r"[a-z0-9]+", _fold(text))


# ---------------------------------------------------------------------------
# the closed metric library — each: (predicted, expected, params) -> float in [0, 1]


def _m_exact(predicted: Any, expected: Any, params: dict) -> float:
    """Plain value equality. In the benchy engine the ``exact`` dimension dispatches
    to ``types.equal`` instead (semantic equality: parsed temporals, byte-compared
    artifacts); this plain form is what the vendored copy uses where no schema node
    is at hand."""
    return 1.0 if predicted == expected else 0.0


def _m_casefold_strip(predicted: Any, expected: Any, params: dict) -> float:
    if not isinstance(predicted, str) or not isinstance(expected, str):
        return 1.0 if predicted == expected else 0.0
    return 1.0 if predicted.strip().casefold() == expected.strip().casefold() else 0.0


def _m_token_set_f1(predicted: Any, expected: Any, params: dict) -> float:
    """F1 over the unordered token sets — reordered-token extractions score 1."""
    mine, theirs = _tokens(predicted), _tokens(expected)
    if not mine or not theirs:
        return 1.0 if mine == theirs else 0.0
    inter = len(mine & theirs)
    return 2 * inter / (len(mine) + len(theirs))


_DATE_PATTERNS = (
    re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$"),   # d/m/y or m/d/y
    re.compile(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$"),     # ISO y-m-d
)

_ES_MONTHS = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5,
              "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9,
              "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}


def _parse_date(text: Any):
    """Best-effort parse of common numeric/Spanish date spellings -> date or None.

    Ambiguous d/m vs m/d resolves to day-first (the locale of our cases); a
    two-digit year maps to 2000+yy. Unparseable is not a match -- conservative.
    """
    if not isinstance(text, str):
        return None
    folded = _fold(text)
    match = _DATE_PATTERNS[0].match(folded)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if year < 100:
            year += 2000
        if day > 31 and month <= 12:  # y/m/d written with separators flipped
            day, month, year = month, day, year
        try:
            return datetime(year, month, day).date()
        except ValueError:
            return None
    match = _DATE_PATTERNS[1].match(folded)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            return datetime(year, month, day).date()
        except ValueError:
            return None
    match = re.match(r"^(\d{1,2})\s+de\s+([a-z]+)\s+(?:de\s+)?(\d{4})$", folded)
    if match and match.group(2) in _ES_MONTHS:
        try:
            return datetime(int(match.group(3)), _ES_MONTHS[match.group(2)],
                            int(match.group(1))).date()
        except ValueError:
            return None
    return None


def _m_date_flexible(predicted: Any, expected: Any, params: dict) -> float:
    """Compare parsed dates, not strings: "12/3/26" matches "2026-03-12"."""
    got, want = _parse_date(predicted), _parse_date(expected)
    return 1.0 if (got is not None and want is not None and got == want) else 0.0


def _as_float(value: Any) -> float | None:
    """Coercion shared with datapipeline's derived_scoring (comma decimals)."""
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def _m_numeric_tolerance(predicted: Any, expected: Any, params: dict) -> float:
    """1 if |predicted - expected| <= params.tolerance (default 0), else 0."""
    tolerance = (params or {}).get("tolerance", 0.0)
    got, want = _as_float(predicted), _as_float(expected)
    if got is None or want is None:
        return 0.0
    return 1.0 if abs(got - want) <= max(float(tolerance), 0.0) else 0.0


def _m_span_recall(predicted: Any, expected: Any, params: dict) -> float:
    """Fraction of the expected's tokens contained in the prediction (from dp).

    The expected side keeps repetitions (a token expected twice and found once
    counts 1/2); the prediction side is a set (finding it once is finding it).
    """
    objetivo = _token_list(expected)
    if not objetivo:
        return 0.0
    texto = set(_token_list(predicted))
    return len([t for t in objetivo if t in texto]) / len(objetivo)


def _m_set_f1(predicted: Any, expected: Any, params: dict) -> float:
    """Set F1 (from dp): collections compare member-wise; scalars compare tokens."""
    sa = {str(x) for x in (predicted or [])} if isinstance(predicted, (list, tuple, set)) else set(_token_list(predicted))
    sb = {str(x) for x in (expected or [])} if isinstance(expected, (list, tuple, set)) else set(_token_list(expected))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return 2 * inter / (len(sa) + len(sb))


#: The registry. Closed on purpose: a benchmark names one of these, nothing else.
METRICS = {
    "exact": _m_exact,
    "casefold_strip": _m_casefold_strip,
    "token_set_f1": _m_token_set_f1,
    "date_flexible": _m_date_flexible,
    "numeric_tolerance": _m_numeric_tolerance,
    "span_recall": _m_span_recall,
    "set_f1": _m_set_f1,
}

#: Metrics that preserve a declared enum vocabulary (a match still lands exactly on
#: a declared label). The rest would dissolve the vocabulary contract.
ENUM_SAFE = frozenset({"exact", "casefold_strip"})

#: Parameter bounds per metric — the compiler rejects anything outside these.
PARAM_SPEC = {
    "exact": {},
    "casefold_strip": {},
    "token_set_f1": {},
    "date_flexible": {},
    "numeric_tolerance": {"tolerance": (0.0, None)},
    "span_recall": {},
    "set_f1": {},
}


def score(name: str, predicted: Any, expected: Any, params: dict | None = None) -> float:
    """One field, one metric, one float in [0, 1]. `name` must be in the registry."""
    fn = METRICS.get(name)
    if fn is None:
        raise ValueError(f"unknown metric {name!r} (registry: {', '.join(sorted(METRICS))})")
    return float(fn(predicted, expected, dict(params or {})))


def catalog() -> dict:
    """What a benchmark author is shown: names, allowed params with bounds, enum-safety."""
    return {name: {"params": {param: [bounds[0], "unbounded"] if bounds[1] is None else list(bounds)
                              for param, bounds in PARAM_SPEC[name].items()},
                   "enum_safe": name in ENUM_SAFE}
            for name in sorted(METRICS)}
