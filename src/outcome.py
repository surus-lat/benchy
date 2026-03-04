"""Run/task outcome helpers for concise machine-readable status reporting."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


TASK_STATUS_PASSED = "passed"
TASK_STATUS_DEGRADED = "degraded"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_SKIPPED = "skipped"
TASK_STATUS_NO_SAMPLES = "no_samples"
TASK_STATUS_ERROR = "error"
TASK_STATUS_PENDING = "pending"

COMPLETED_TASK_STATUSES = {
    TASK_STATUS_PASSED,
    TASK_STATUS_DEGRADED,
    TASK_STATUS_SKIPPED,
    TASK_STATUS_NO_SAMPLES,
}


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def summarize_metrics(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract/derive common status-relevant metrics from aggregate metrics."""
    total_samples = _coerce_int(metrics.get("total_samples"))
    if total_samples is None:
        total_samples = _coerce_int(metrics.get("run_samples"))

    valid_samples = _coerce_int(metrics.get("valid_samples"))
    error_count = _coerce_int(metrics.get("error_count"))
    error_rate = _coerce_float(metrics.get("error_rate"))
    invalid_response_rate = _coerce_float(metrics.get("invalid_response_rate"))
    connectivity_error_rate = _coerce_float(metrics.get("connectivity_error_rate"))
    response_rate = _coerce_float(metrics.get("response_rate"))

    if valid_samples is None and total_samples is not None and error_count is not None:
        valid_samples = max(total_samples - error_count, 0)
    if error_count is None and total_samples is not None and valid_samples is not None:
        error_count = max(total_samples - valid_samples, 0)
    if error_rate is None and total_samples and error_count is not None:
        error_rate = float(error_count) / float(total_samples)

    return {
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "error_count": error_count,
        "error_rate": error_rate,
        "invalid_response_rate": invalid_response_rate,
        "connectivity_error_rate": connectivity_error_rate,
        "response_rate": response_rate,
    }


def evaluate_metric_status(metrics: Dict[str, Any]) -> Tuple[str, Optional[str], Dict[str, Optional[float]]]:
    """Classify a single metrics payload into a status."""
    summary = summarize_metrics(metrics)
    total_samples = summary.get("total_samples")
    valid_samples = summary.get("valid_samples")
    error_rate = summary.get("error_rate")
    invalid_response_rate = summary.get("invalid_response_rate")
    connectivity_error_rate = summary.get("connectivity_error_rate")

    if total_samples == 0:
        return TASK_STATUS_NO_SAMPLES, "no_samples", summary

    if total_samples is not None and total_samples > 0 and valid_samples == 0:
        if connectivity_error_rate == 1.0:
            return TASK_STATUS_FAILED, "all_connectivity_errors", summary
        if invalid_response_rate == 1.0:
            return TASK_STATUS_FAILED, "all_invalid_responses", summary
        return TASK_STATUS_FAILED, "no_valid_samples", summary

    if error_rate is not None and error_rate >= 1.0:
        return TASK_STATUS_FAILED, "all_samples_failed", summary

    has_partial_errors = False
    for value in (error_rate, invalid_response_rate, connectivity_error_rate):
        if value is not None and value > 0.0:
            has_partial_errors = True
            break

    if has_partial_errors:
        return TASK_STATUS_DEGRADED, "partial_errors", summary

    return TASK_STATUS_PASSED, None, summary


def build_task_outcome(
    *,
    task_name: str,
    aggregated_metrics: Dict[str, Any],
    subtask_metrics: Dict[str, Dict[str, Any]],
    skipped_subtasks: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Build task-level outcome from per-subtask metrics and skip info."""
    skipped_map = {
        item.get("subtask"): item.get("reason", "incompatible")
        for item in (skipped_subtasks or [])
        if isinstance(item, dict) and item.get("subtask")
    }

    subtask_names = set(subtask_metrics.keys()) | set(skipped_map.keys())
    subtask_outcomes: Dict[str, Dict[str, Any]] = {}

    for subtask_name in sorted(subtask_names):
        if subtask_name in skipped_map:
            subtask_outcomes[subtask_name] = {
                "status": TASK_STATUS_SKIPPED,
                "reason": skipped_map[subtask_name],
                "summary": summarize_metrics({}),
            }
            continue

        metrics = subtask_metrics.get(subtask_name) or {}
        status, reason, summary = evaluate_metric_status(metrics)
        subtask_outcomes[subtask_name] = {
            "status": status,
            "reason": reason,
            "summary": summary,
        }

    task_status, task_reason, task_summary = evaluate_metric_status(aggregated_metrics or {})
    if not subtask_outcomes and not aggregated_metrics:
        task_status, task_reason = TASK_STATUS_NO_SAMPLES, "no_metrics"
    if subtask_outcomes:
        subtask_statuses = [entry["status"] for entry in subtask_outcomes.values()]
        if any(status in {TASK_STATUS_FAILED, TASK_STATUS_ERROR} for status in subtask_statuses):
            task_status, task_reason = TASK_STATUS_FAILED, "subtask_failure"
        elif any(status == TASK_STATUS_SKIPPED for status in subtask_statuses):
            task_status, task_reason = TASK_STATUS_SKIPPED, "subtask_skipped"
        elif any(status == TASK_STATUS_NO_SAMPLES for status in subtask_statuses):
            task_status, task_reason = TASK_STATUS_NO_SAMPLES, "subtask_no_samples"
        elif any(status == TASK_STATUS_DEGRADED for status in subtask_statuses):
            task_status, task_reason = TASK_STATUS_DEGRADED, "subtask_degraded"
        else:
            task_status, task_reason = TASK_STATUS_PASSED, None

    return {
        "task": task_name,
        "status": task_status,
        "reason": task_reason,
        "summary": task_summary,
        "subtasks": subtask_outcomes,
        "updated_at": datetime.now().isoformat(),
    }


def resolve_exit_code(task_statuses: List[str], *, policy: str) -> int:
    """Compute process exit code from task statuses and policy."""
    normalized = [status or TASK_STATUS_PENDING for status in task_statuses]
    policy = (policy or "relaxed").lower()

    if policy == "relaxed":
        return 0

    if policy == "smoke":
        blocking = {
            TASK_STATUS_FAILED,
            TASK_STATUS_ERROR,
            TASK_STATUS_PENDING,
            TASK_STATUS_SKIPPED,
            TASK_STATUS_NO_SAMPLES,
        }
        return 1 if any(status in blocking for status in normalized) else 0

    if policy == "strict":
        return 1 if any(status != TASK_STATUS_PASSED for status in normalized) else 0

    return 0


def resolve_run_status(task_statuses: List[str]) -> str:
    """Map task statuses to an overall run status."""
    normalized = [status or TASK_STATUS_PENDING for status in task_statuses]
    if any(status in {TASK_STATUS_FAILED, TASK_STATUS_ERROR, TASK_STATUS_PENDING} for status in normalized):
        return TASK_STATUS_FAILED
    if any(status in {TASK_STATUS_DEGRADED, TASK_STATUS_SKIPPED, TASK_STATUS_NO_SAMPLES} for status in normalized):
        return TASK_STATUS_DEGRADED
    return TASK_STATUS_PASSED
