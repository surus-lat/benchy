"""Field-level diagnostics reporting for structured extraction handlers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _truncate_text(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return value
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3] + "..."


def _compact_value(value: Any, max_value_chars: int) -> Any:
    if isinstance(value, str):
        return _truncate_text(value, max_value_chars)
    if value is None or isinstance(value, (int, float, bool)):
        return value

    try:
        dumped = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        dumped = _truncate_text(str(value), max_value_chars)
    return _truncate_text(dumped, max_value_chars)


def _classify_field_error(match_type: str, type_match: bool) -> Optional[str]:
    if match_type == "missed":
        return "missing_field"
    if match_type == "spurious":
        return "spurious_field"
    if match_type == "partial":
        return "partial_match"
    if match_type == "incorrect":
        return "type_mismatch" if not type_match else "incorrect_value"
    return None


def build_field_diagnostics_report(
    *,
    per_sample_metrics: List[Dict[str, Any]],
    max_examples_per_field: int = 20,
    max_fields_in_report: int = 200,
    max_value_chars: int = 240,
    require_single_schema: bool = True,
) -> Dict[str, Any]:
    """Build a field-level diagnostics report from per-sample metrics."""
    metrics_with_fields = [
        m for m in per_sample_metrics
        if isinstance(m, dict) and isinstance(m.get("field_results"), dict) and m.get("field_results")
    ]

    if not metrics_with_fields:
        return {"status": "skipped", "reason": "no_field_results"}

    schema_fingerprints = sorted({
        str(m.get("schema_fingerprint"))
        for m in metrics_with_fields
        if m.get("schema_fingerprint")
    })
    if require_single_schema and len(schema_fingerprints) > 1:
        return {
            "status": "skipped",
            "reason": "multiple_schema_fingerprints",
            "schema_fingerprints": schema_fingerprints,
        }

    field_stats: Dict[str, Dict[str, Any]] = {}
    for metric in metrics_with_fields:
        sample_id = metric.get("sample_id")
        field_results = metric.get("field_results", {})
        for field_key, result in field_results.items():
            stats = field_stats.setdefault(
                str(field_key),
                {
                    "field": str(field_key),
                    "expected_count": 0,
                    "predicted_count": 0,
                    "exact_count": 0,
                    "partial_count": 0,
                    "incorrect_count": 0,
                    "missing_count": 0,
                    "spurious_count": 0,
                    "type_mismatch_count": 0,
                    "error_type_counts": Counter(),
                    "examples_by_error_type": defaultdict(list),
                },
            )

            match_type = str(result.get("match_type") or "unknown")
            type_match_raw = result.get("type_match")
            type_match = type_match_raw if isinstance(type_match_raw, bool) else True
            expected_value = result.get("expected")
            predicted_value = result.get("predicted")

            if match_type != "spurious":
                stats["expected_count"] += 1
            if match_type != "missed":
                stats["predicted_count"] += 1

            if match_type == "exact":
                stats["exact_count"] += 1
            elif match_type == "partial":
                stats["partial_count"] += 1
            elif match_type == "incorrect":
                stats["incorrect_count"] += 1
            elif match_type == "missed":
                stats["missing_count"] += 1
            elif match_type == "spurious":
                stats["spurious_count"] += 1

            if not type_match and match_type in {"exact", "partial", "incorrect"}:
                stats["type_mismatch_count"] += 1

            error_type = _classify_field_error(match_type, type_match)
            if not error_type:
                continue

            stats["error_type_counts"][error_type] += 1
            examples = stats["examples_by_error_type"][error_type]
            if len(examples) < max_examples_per_field:
                examples.append(
                    {
                        "sample_id": sample_id,
                        "expected": _compact_value(expected_value, max_value_chars),
                        "predicted": _compact_value(predicted_value, max_value_chars),
                    }
                )

    fields: List[Dict[str, Any]] = []
    for field_key, stats in field_stats.items():
        expected_count = int(stats["expected_count"])
        predicted_count = int(stats["predicted_count"])
        exact_count = int(stats["exact_count"])
        partial_count = int(stats["partial_count"])
        incorrect_count = int(stats["incorrect_count"])
        missing_count = int(stats["missing_count"])
        spurious_count = int(stats["spurious_count"])
        type_mismatch_count = int(stats["type_mismatch_count"])

        if expected_count > 0:
            exact_accuracy = exact_count / expected_count
            error_rate = (partial_count + incorrect_count + missing_count) / expected_count
        else:
            exact_accuracy = None
            error_rate = None

        spurious_rate = spurious_count / predicted_count if predicted_count > 0 else 0.0

        fields.append(
            {
                "field": field_key,
                "expected_count": expected_count,
                "predicted_count": predicted_count,
                "exact_count": exact_count,
                "partial_count": partial_count,
                "incorrect_count": incorrect_count,
                "missing_count": missing_count,
                "spurious_count": spurious_count,
                "type_mismatch_count": type_mismatch_count,
                "exact_accuracy": exact_accuracy,
                "error_rate": error_rate,
                "spurious_rate": spurious_rate,
                "error_type_counts": dict(stats["error_type_counts"]),
                "examples_by_error_type": {
                    key: list(value)
                    for key, value in stats["examples_by_error_type"].items()
                },
            }
        )

    expected_fields = [f for f in fields if f["expected_count"] > 0]
    expected_fields.sort(
        key=lambda item: (
            item["exact_accuracy"] if item["exact_accuracy"] is not None else 1.0,
            -item["expected_count"],
            item["field"],
        )
    )
    hallucinated_fields = [f for f in fields if f["expected_count"] == 0 and f["spurious_count"] > 0]
    hallucinated_fields.sort(key=lambda item: (-item["spurious_count"], item["field"]))

    capped_fields = expected_fields[:max_fields_in_report]
    omitted_fields = max(0, len(expected_fields) - len(capped_fields))
    capped_hallucinated_fields = hallucinated_fields[:max_fields_in_report]
    omitted_hallucinated_fields = max(0, len(hallucinated_fields) - len(capped_hallucinated_fields))

    total_expected_observations = sum(f["expected_count"] for f in expected_fields)
    total_exact_observations = sum(f["exact_count"] for f in expected_fields)
    total_error_observations = sum(
        f["partial_count"] + f["incorrect_count"] + f["missing_count"] for f in expected_fields
    )
    total_spurious_observations = sum(f["spurious_count"] for f in fields)

    return {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "samples_with_field_results": len(metrics_with_fields),
            "fields_analyzed": len(expected_fields),
            "hallucinated_fields": len(hallucinated_fields),
            "total_expected_observations": total_expected_observations,
            "total_exact_observations": total_exact_observations,
            "total_error_observations": total_error_observations,
            "total_spurious_observations": total_spurious_observations,
            "exact_accuracy_overall": (
                total_exact_observations / total_expected_observations if total_expected_observations > 0 else 0.0
            ),
            "max_examples_per_field": max_examples_per_field,
        },
        "schema": {
            "fingerprint": schema_fingerprints[0] if len(schema_fingerprints) == 1 else None,
            "fingerprints_seen": schema_fingerprints,
        },
        "fields": capped_fields,
        "hallucinated_only_fields": capped_hallucinated_fields,
        "omitted_fields": omitted_fields,
        "omitted_hallucinated_fields": omitted_hallucinated_fields,
    }


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def render_field_diagnostics_text(report: Dict[str, Any], examples_per_field: int = 5) -> str:
    """Render a concise plain-text report from field diagnostics JSON."""
    if report.get("status") != "ok":
        reason = report.get("reason", "unknown")
        return f"Field diagnostics skipped: {reason}\n"

    summary = report.get("summary", {})
    schema_info = report.get("schema", {})
    lines: List[str] = []

    lines.append("=" * 60)
    lines.append("FIELD DIAGNOSTICS REPORT")
    lines.append("=" * 60)
    lines.append(f"Samples analyzed: {summary.get('samples_with_field_results', 0)}")
    lines.append(f"Schema fingerprint: {schema_info.get('fingerprint') or 'n/a'}")
    lines.append(f"Fields analyzed: {summary.get('fields_analyzed', 0)}")
    lines.append(f"Overall exact accuracy: {_pct(summary.get('exact_accuracy_overall'))}")
    lines.append("")
    lines.append("WORST FIELDS (LOWEST EXACT ACCURACY)")
    lines.append("-" * 60)

    fields = report.get("fields", []) or []
    if not fields:
        lines.append("No field-level diagnostics available.")
    else:
        for field in fields:
            lines.append(
                f"{field['field']}: exact={_pct(field.get('exact_accuracy'))} "
                f"(exact={field.get('exact_count', 0)}/{field.get('expected_count', 0)}), "
                f"partial={field.get('partial_count', 0)}, "
                f"incorrect={field.get('incorrect_count', 0)}, "
                f"missing={field.get('missing_count', 0)}, "
                f"type_mismatch={field.get('type_mismatch_count', 0)}"
            )

            examples_by_error_type = field.get("examples_by_error_type", {}) or {}
            emitted = 0
            for error_type in ("type_mismatch", "incorrect_value", "partial_match", "missing_field", "spurious_field"):
                examples = examples_by_error_type.get(error_type, [])
                if not examples:
                    continue
                for example in examples[:examples_per_field]:
                    lines.append(
                        f"  - {error_type} | sample={example.get('sample_id')} | "
                        f"expected={example.get('expected')} | predicted={example.get('predicted')}"
                    )
                    emitted += 1
                    if emitted >= examples_per_field:
                        break
                if emitted >= examples_per_field:
                    break
            lines.append("")

    hallucinated = report.get("hallucinated_only_fields", []) or []
    if hallucinated:
        lines.append("HALLUCINATED-ONLY FIELDS")
        lines.append("-" * 60)
        for field in hallucinated:
            lines.append(f"{field['field']}: spurious_count={field.get('spurious_count', 0)}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines) + "\n"


def write_field_diagnostics_artifacts(
    *,
    report: Dict[str, Any],
    output_dir: Path,
    safe_model_name: str,
    timestamp: str,
) -> List[Path]:
    """Write field diagnostics artifacts to disk."""
    if report.get("status") != "ok":
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{safe_model_name}_{timestamp}_field_diagnostics.json"
    txt_path = output_dir / f"{safe_model_name}_{timestamp}_field_diagnostics.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(render_field_diagnostics_text(report))

    return [json_path, txt_path]
