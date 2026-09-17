from src.outcome import (
    TASK_STATUS_DEGRADED,
    TASK_STATUS_FAILED,
    TASK_STATUS_NO_SAMPLES,
    TASK_STATUS_PASSED,
    TASK_STATUS_SKIPPED,
    build_task_outcome,
    evaluate_metric_status,
    resolve_exit_code,
    resolve_run_status,
    summarize_metrics,
)


def test_summarize_metrics_derives_error_fields() -> None:
    summary = summarize_metrics({"total_samples": 10, "valid_samples": 7})
    assert summary["error_count"] == 3
    assert summary["error_rate"] == 0.3


def test_evaluate_metric_status_no_samples() -> None:
    status, reason, summary = evaluate_metric_status({"total_samples": 0})
    assert status == TASK_STATUS_NO_SAMPLES
    assert reason == "no_samples"
    assert summary["total_samples"] == 0


def test_evaluate_metric_status_passed() -> None:
    status, reason, _ = evaluate_metric_status({"total_samples": 5, "valid_samples": 5, "error_rate": 0.0})
    assert status == TASK_STATUS_PASSED
    assert reason is None


def test_evaluate_metric_status_degraded_on_partial_errors() -> None:
    status, reason, _ = evaluate_metric_status({"total_samples": 10, "valid_samples": 8, "error_rate": 0.2})
    assert status == TASK_STATUS_DEGRADED
    assert reason == "partial_errors"


def test_build_task_outcome_prioritizes_subtask_failure() -> None:
    outcome = build_task_outcome(
        task_name="document_extraction",
        aggregated_metrics={"total_samples": 10, "valid_samples": 10, "error_rate": 0.0},
        subtask_metrics={
            "ok_subtask": {"total_samples": 2, "valid_samples": 2, "error_rate": 0.0},
            "bad_subtask": {
                "total_samples": 2,
                "valid_samples": 0,
                "connectivity_error_rate": 1.0,
            },
        },
        skipped_subtasks=[],
    )

    assert outcome["status"] == TASK_STATUS_FAILED
    assert outcome["reason"] == "subtask_failure"
    assert outcome["subtasks"]["ok_subtask"]["status"] == TASK_STATUS_PASSED
    assert outcome["subtasks"]["bad_subtask"]["status"] == TASK_STATUS_FAILED


def test_build_task_outcome_marks_skipped_subtasks() -> None:
    outcome = build_task_outcome(
        task_name="spanish",
        aggregated_metrics={"total_samples": 1, "valid_samples": 1},
        subtask_metrics={},
        skipped_subtasks=[{"subtask": "xnli", "reason": "incompatible"}],
    )

    assert outcome["status"] == TASK_STATUS_SKIPPED
    assert outcome["reason"] == "subtask_skipped"
    assert outcome["subtasks"]["xnli"]["status"] == TASK_STATUS_SKIPPED


def test_resolve_exit_code_policies() -> None:
    assert resolve_exit_code([TASK_STATUS_PASSED], policy="relaxed") == 0
    assert resolve_exit_code([TASK_STATUS_PASSED, TASK_STATUS_DEGRADED], policy="smoke") == 0
    assert resolve_exit_code([TASK_STATUS_SKIPPED], policy="smoke") == 1
    assert resolve_exit_code([TASK_STATUS_PASSED, TASK_STATUS_DEGRADED], policy="strict") == 1


def test_resolve_run_status() -> None:
    assert resolve_run_status([TASK_STATUS_PASSED]) == TASK_STATUS_PASSED
    assert resolve_run_status([TASK_STATUS_DEGRADED]) == TASK_STATUS_DEGRADED
    assert resolve_run_status([TASK_STATUS_SKIPPED]) == TASK_STATUS_DEGRADED
    assert resolve_run_status([TASK_STATUS_FAILED]) == TASK_STATUS_FAILED
