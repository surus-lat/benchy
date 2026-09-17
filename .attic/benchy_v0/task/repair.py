"""Best-effort JSON extraction and repair from model prose.

A system without native structured output has the JSON Schema stuffed into
its prompt and is asked, in English, to reply with JSON only. Real systems
do not reliably comply: they wrap the answer in a fenced code block, add a
sentence before or after it, use single quotes, leave a trailing comma,
emit a literal newline inside a string, or occasionally just refuse. This
module turns "prose with JSON somewhere in it, possibly slightly malformed"
into a Python value -- or explains why it could not -- and never raises.
Every call returns a ``(value, error)`` pair; exactly one side is ``None``.

The strategy, in order:

1. Collect *candidate* substrings that might be the JSON payload: the
   full text, every fenced code block (```json ... ``` or ``` ... ```,
   in order of appearance), and every top-level balanced ``{...}``/``[...]``
   span found by a string-aware brace scanner (so braces embedded inside
   string literals don't throw off the count -- the naive
   ``text.find("{")``/``text.rfind("}")`` approach from the salvaged
   ``openai_interface._extract_json`` breaks on nested objects with braces
   in string values).
2. For each candidate, try ``json.loads`` directly, then progressively
   apply repairs (trailing commas, unquoted keys, Python literals
   True/False/None, unescaped control characters inside strings, and
   finally single-to-double-quote conversion as a last resort) and retry.
3. First success wins. If nothing parses, return a descriptive error.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)\n?```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_UNQUOTED_KEY_RE = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:)')
_PY_LITERALS = (
    (re.compile(r"\bTrue\b"), "true"),
    (re.compile(r"\bFalse\b"), "false"),
    (re.compile(r"\bNone\b"), "null"),
    (re.compile(r"\bNaN\b"), "null"),
)


def _balanced_spans(text: str) -> list[str]:
    """Every top-level ``{...}``/``[...]`` span, respecting string literals."""
    spans: list[str] = []
    stack: list[str] = []
    start: int | None = None
    in_string = False
    quote = ""
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_string = False
            continue
        if ch in "\"'":
            in_string = True
            quote = ch
            continue
        if ch in "{[":
            if not stack:
                start = i
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    spans.append(text[start : i + 1])
                    start = None
    return spans


def _fix_unescaped_control_chars(candidate: str) -> str:
    """Escape raw newline/tab/CR bytes that landed inside a JSON string."""
    out: list[str] = []
    in_string = False
    escape = False
    for ch in candidate:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                in_string = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
            continue
        if ch == '"':
            in_string = True
        out.append(ch)
    return "".join(out)


def _single_to_double_quotes(candidate: str) -> str:
    """Last-resort repair for Python-dict-style output (single-quoted keys).

    Only ever tried after a direct parse has already failed, so well-formed
    JSON containing an apostrophe (``"O'Brien"``) is never touched -- it
    would have parsed on an earlier attempt.
    """
    if "'" not in candidate:
        return candidate
    out: list[str] = []
    in_single = False
    in_double = False
    escape = False
    for ch in candidate:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            continue
        if in_double:
            out.append(ch)
            if ch == '"':
                in_double = False
            continue
        if in_single:
            if ch == "'":
                in_single = False
                out.append('"')
            elif ch == '"':
                out.append('\\"')
            else:
                out.append(ch)
            continue
        if ch == '"':
            in_double = True
            out.append(ch)
        elif ch == "'":
            in_single = True
            out.append('"')
        else:
            out.append(ch)
    return "".join(out)


def _quote_unquoted_keys(candidate: str) -> str:
    return _UNQUOTED_KEY_RE.sub(lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}', candidate)


def _repair_variants(candidate: str):
    """Yield progressively more aggressive repairs of `candidate`."""
    yield candidate
    no_commas = _TRAILING_COMMA_RE.sub(r"\1", candidate)
    yield no_commas
    quoted_keys = _quote_unquoted_keys(no_commas)
    yield quoted_keys
    literals = quoted_keys
    for pattern, repl in _PY_LITERALS:
        literals = pattern.sub(repl, literals)
    yield literals
    fixed_ctrl = _fix_unescaped_control_chars(literals)
    yield fixed_ctrl
    yield _single_to_double_quotes(fixed_ctrl)


def _candidates(text: str) -> list[str]:
    seen: list[str] = []

    def add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.append(candidate)

    add(text)
    for match in _FENCE_RE.finditer(text):
        add(match.group(1))
    for span in _balanced_spans(text):
        add(span)
    return seen


def extract_json(text: str | None) -> tuple[Any | None, str | None]:
    """Best-effort JSON extraction. Returns ``(value, None)`` or ``(None, error)``.

    Never raises -- a benchmark run must survive one bad response.
    """
    if text is None:
        return None, "response has no text to parse"
    stripped = text.strip()
    if not stripped:
        return None, "response text is empty"

    last_error = "no JSON-shaped content found in response text"
    for candidate in _candidates(stripped):
        for repaired in _repair_variants(candidate):
            try:
                return json.loads(repaired), None
            except Exception as exc:  # noqa: BLE001 - repair must never raise
                last_error = f"{type(exc).__name__}: {exc}"
    return None, f"could not parse JSON from response text ({last_error})"
