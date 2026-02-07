import pytest

from src.engine.retry import (
    RetryDecision,
    RetryableError,
    classify_http_exception,
    exponential_backoff,
    extract_status_code,
    is_rate_limit_error,
    rate_limit_backoff,
    run_with_retries,
)


class _ExcWithStatus(Exception):
    def __init__(self, status_code: int, message: str = "error") -> None:
        super().__init__(message)
        self.status_code = status_code


def test_backoff_helpers_cap_values() -> None:
    assert exponential_backoff(1, base_delay=1.0, multiplier=2.0, max_delay=16.0) == 1.0
    assert exponential_backoff(10, base_delay=1.0, multiplier=2.0, max_delay=16.0) == 16.0
    assert rate_limit_backoff(1, base_delay=5.0, multiplier=3.0, max_delay=120.0) == 5.0
    assert rate_limit_backoff(5, base_delay=5.0, multiplier=3.0, max_delay=120.0) == 120.0


def test_extract_status_code_and_rate_limit_detection() -> None:
    exc = _ExcWithStatus(429, "rate limit")
    assert extract_status_code(exc) == 429
    assert is_rate_limit_error(exc) is True


def test_classify_http_exception_4xx_no_retry_by_default() -> None:
    decision = classify_http_exception(_ExcWithStatus(400, "bad request"), attempt=1)
    assert decision == RetryDecision(retry=False, delay=0.0, error_type="invalid_response")


def test_classify_http_exception_4xx_can_retry_with_flag() -> None:
    decision = classify_http_exception(_ExcWithStatus(404, "not found"), attempt=1, retry_on_4xx=True)
    assert decision.retry is True
    assert decision.error_type == "invalid_response"


def test_classify_http_exception_5xx_retries_as_connectivity() -> None:
    decision = classify_http_exception(_ExcWithStatus(503, "service unavailable"), attempt=2)
    assert decision.retry is True
    assert decision.error_type == "connectivity_error"
    assert decision.delay > 0


@pytest.mark.asyncio
async def test_run_with_retries_succeeds_after_retryable_error() -> None:
    state = {"calls": 0}

    async def attempt_fn(_: int) -> str:
        state["calls"] += 1
        if state["calls"] == 1:
            raise RetryableError("try again", error_type="connectivity_error", retry_after=0.0)
        return "ok"

    result, error, error_type = await run_with_retries(
        attempt_fn,
        max_retries=3,
        classify_error=lambda exc, attempt: RetryDecision(True, 0.0, "connectivity_error"),
    )

    assert result == "ok"
    assert error is None
    assert error_type is None
    assert state["calls"] == 2


@pytest.mark.asyncio
async def test_run_with_retries_stops_when_classifier_disables_retry() -> None:
    async def attempt_fn(_: int) -> str:
        raise ValueError("fatal")

    result, error, error_type = await run_with_retries(
        attempt_fn,
        max_retries=5,
        classify_error=lambda exc, attempt: RetryDecision(False, 0.0, "invalid_response"),
    )

    assert result is None
    assert error == "fatal"
    assert error_type == "invalid_response"
