"""Task-aware output diagnostics for benchmark runs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from jsonschema import ValidationError, validate

from ..common.schema_sanitizer import unwrap_openai_response_format_schema


def _extract_json_candidate(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text

    patterns = [
        r"```json\s*\n(.*?)\n```",
        r"```\s*\n(.*?)\n```",
        r"```json(.*?)```",
        r"```(.*?)```",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        if candidate.count("{") == candidate.count("}"):
            return candidate
    return text


def _max_whitespace_run(text: str) -> int:
    max_run = 0
    run = 0
    for ch in text:
        if ch in (" ", "\n", "\r", "\t"):
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run


def _detect_repetition(text: str) -> bool:
    if not text:
        return False
    if re.search(r"\s{80,}", text):
        return True
    tokens = text.split()
    if len(tokens) < 18:
        return False
    for i in range(0, len(tokens) - 15):
        tri = tokens[i:i + 3]
        if len(set(tri)) == 1:
            return True
        if tokens[i:i + 15].count(tri[0]) >= 6:
            return True
    return False


def _looks_like_markup(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"<(table|tr|td|th|html|body|div|span)\b", text, re.IGNORECASE))


def _validate_schema(payload: Any, sample_schema: Optional[Dict[str, Any]]) -> bool:
    schema = unwrap_openai_response_format_schema(sample_schema or {})
    if not isinstance(schema, dict) or not schema:
        return False
    try:
        validate(instance=payload, schema=schema)
        return True
    except ValidationError:
        return False


def _analyze_structured(
    *,
    sample: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    raw_text = output.get("raw")
    payload = output.get("output")
    parse_ok = False
    schema_valid = False
    shape = "plain_text"

    if payload is not None:
        parse_ok = True
        schema_valid = _validate_schema(payload, sample.get("schema"))
        shape = "valid_json_schema" if schema_valid else "valid_json_non_schema"
    else:
        text = raw_text if isinstance(raw_text, str) else ""
        cleaned = _extract_json_candidate(text)
        try:
            parsed = json.loads(cleaned)
            parse_ok = True
            schema_valid = _validate_schema(parsed, sample.get("schema"))
            shape = "valid_json_schema" if schema_valid else "valid_json_non_schema"
        except Exception:
            if _looks_like_markup(text):
                shape = "ocr_text_or_markup"
            elif text.count("{") > text.count("}") and text.strip().endswith((" ", "\t", "\n", "\r", ",", "{", "[")):
                shape = "json_truncated"
            elif _detect_repetition(text):
                shape = "degenerate_repetition"
            else:
                shape = "json_parse_error"

    text = raw_text if isinstance(raw_text, str) else ""
    return {
        "diagnostic_class": shape,
        "json_parse_ok": parse_ok,
        "schema_valid": schema_valid,
        "max_whitespace_run": _max_whitespace_run(text),
        "repetition_detected": _detect_repetition(text),
    }


def _analyze_multiple_choice(*, sample: Dict[str, Any], output: Dict[str, Any]) -> Dict[str, Any]:
    pred = output.get("output")
    choices = sample.get("choices") or []
    valid_choice = isinstance(pred, int) and pred >= 0 and pred < len(choices)
    cls = "mc_valid" if valid_choice else "mc_invalid"
    return {
        "diagnostic_class": cls,
        "json_parse_ok": None,
        "schema_valid": None,
        "max_whitespace_run": 0,
        "repetition_detected": False,
    }


def _analyze_generic(*, output: Dict[str, Any]) -> Dict[str, Any]:
    raw_text = output.get("raw")
    text = raw_text if isinstance(raw_text, str) else ""
    if not text:
        return {
            "diagnostic_class": "not_applicable",
            "json_parse_ok": None,
            "schema_valid": None,
            "max_whitespace_run": 0,
            "repetition_detected": False,
        }
    if _looks_like_markup(text):
        cls = "ocr_text_or_markup"
    elif _detect_repetition(text):
        cls = "degenerate_repetition"
    else:
        cls = "plain_text"
    return {
        "diagnostic_class": cls,
        "json_parse_ok": None,
        "schema_valid": None,
        "max_whitespace_run": _max_whitespace_run(text),
        "repetition_detected": _detect_repetition(text),
    }


def analyze_output(
    *,
    sample: Dict[str, Any],
    output: Dict[str, Any],
    task: Any,
) -> Dict[str, Any]:
    """Return task-aware output diagnostics for one sample."""
    answer_type = getattr(task, "answer_type", None)

    if answer_type == "structured" or sample.get("schema") is not None:
        diagnostics = _analyze_structured(sample=sample, output=output)
    elif answer_type == "multiple_choice":
        diagnostics = _analyze_multiple_choice(sample=sample, output=output)
    else:
        diagnostics = _analyze_generic(output=output)

    diagnostics.update(
        {
            "finish_reason": output.get("finish_reason"),
            "completion_tokens": output.get("completion_tokens"),
            "prompt_tokens": output.get("prompt_tokens"),
            "raw_length": len(output.get("raw")) if isinstance(output.get("raw"), str) else 0,
        }
    )
    return diagnostics


def aggregate_diagnostics(
    diagnostics_entries: List[Dict[str, Any]],
    *,
    run_samples: int,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    counts: Dict[str, int] = {}
    for item in diagnostics_entries:
        cls = item.get("diagnostic_class")
        if not isinstance(cls, str):
            continue
        counts[cls] = counts.get(cls, 0) + 1

    rates = {
        f"{key}_rate": (value / run_samples if run_samples > 0 else 0.0)
        for key, value in counts.items()
    }
    return counts, rates

