"""Tests for structured field diagnostics report generation."""

from src.tasks.common.utils.field_diagnostics_report import build_field_diagnostics_report


def test_build_field_diagnostics_report_aggregates_field_stats():
    per_sample_metrics = [
        {
            "sample_id": "s1",
            "schema_fingerprint": "abc123",
            "field_results": {
                "total": {
                    "match_type": "exact",
                    "type_match": True,
                    "expected": 100,
                    "predicted": 100,
                },
                "issue_date": {
                    "match_type": "incorrect",
                    "type_match": False,
                    "expected": "2026-01-01",
                    "predicted": 20260101,
                },
                "unexpected_key": {
                    "match_type": "spurious",
                    "type_match": False,
                    "expected": None,
                    "predicted": "noise",
                },
            },
        },
        {
            "sample_id": "s2",
            "schema_fingerprint": "abc123",
            "field_results": {
                "total": {
                    "match_type": "partial",
                    "type_match": True,
                    "expected": 100,
                    "predicted": "100.00",
                },
                "issue_date": {
                    "match_type": "missed",
                    "type_match": False,
                    "expected": "2026-01-02",
                    "predicted": None,
                },
            },
        },
    ]

    report = build_field_diagnostics_report(per_sample_metrics=per_sample_metrics, max_examples_per_field=20)

    assert report["status"] == "ok"
    assert report["schema"]["fingerprint"] == "abc123"
    assert report["summary"]["samples_with_field_results"] == 2

    by_field = {entry["field"]: entry for entry in report["fields"]}
    assert by_field["total"]["expected_count"] == 2
    assert by_field["total"]["exact_count"] == 1
    assert by_field["total"]["partial_count"] == 1
    assert by_field["total"]["exact_accuracy"] == 0.5

    assert by_field["issue_date"]["incorrect_count"] == 1
    assert by_field["issue_date"]["missing_count"] == 1
    assert by_field["issue_date"]["type_mismatch_count"] == 1
    assert by_field["issue_date"]["error_type_counts"]["type_mismatch"] == 1
    assert by_field["issue_date"]["error_type_counts"]["missing_field"] == 1

    hallucinated = {entry["field"]: entry for entry in report["hallucinated_only_fields"]}
    assert hallucinated["unexpected_key"]["spurious_count"] == 1


def test_build_field_diagnostics_report_skips_when_multiple_schemas():
    per_sample_metrics = [
        {"sample_id": "s1", "schema_fingerprint": "schema_a", "field_results": {"a": {"match_type": "exact"}}},
        {"sample_id": "s2", "schema_fingerprint": "schema_b", "field_results": {"a": {"match_type": "exact"}}},
    ]

    report = build_field_diagnostics_report(per_sample_metrics=per_sample_metrics, max_examples_per_field=20)

    assert report["status"] == "skipped"
    assert report["reason"] == "multiple_schema_fingerprints"


def test_build_field_diagnostics_report_skips_without_field_results():
    report = build_field_diagnostics_report(per_sample_metrics=[], max_examples_per_field=20)
    assert report["status"] == "skipped"
    assert report["reason"] == "no_field_results"
