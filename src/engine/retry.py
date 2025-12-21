"""Shared retry helpers for engine and interfaces."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Tuple, TypeVar

import httpx

T = TypeVar("T")


@dataclass(frozen=True)
class RetryDecision:
    """Decision for whether to retry and how long to wait."""

    retry: bool
    delay: float
    error_type: str


class RetryableError(Exception):
    """Error that should be retried with an optional delay."""

    def __init__(
        self,
        message: str,
        error_type: str = "connectivity_error",
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.retry_after = retry_after


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    multiplier: float = 2.0,
    max_delay: float = 16.0,
) -> float:
    """Compute exponential backoff delay for a 1-based attempt."""
    return min(base_delay * (multiplier ** (attempt - 1)), max_delay)


def rate_limit_backoff(
    attempt: int,
    base_delay: float = 5.0,
    multiplier: float = 3.0,
    max_delay: float = 120.0,
) -> float:
    """Compute backoff delay for rate limit retries."""
    return min(base_delay * (multiplier ** (attempt - 1)), max_delay)


def extract_status_code(exc: Exception) -> Optional[int]:
    """Extract a status code from common exception shapes."""
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


def is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort check for rate limit errors."""
    status_code = extract_status_code(exc)
    if status_code == 429:
        return True
    message = str(exc).lower()
    return "rate_limit" in message or "rate limit" in message or "429" in message


def classify_http_exception(
    exc: Exception,
    attempt: int,
    *,
    retry_on_4xx: bool = False,
    base_delay: float = 1.0,
    max_delay: float = 16.0,
) -> RetryDecision:
    """Classify HTTP/network exceptions into retry decisions."""
    status_code = extract_status_code(exc)

    if is_rate_limit_error(exc):
        return RetryDecision(True, rate_limit_backoff(attempt), "rate_limit")

    if status_code is not None:
        if 400 <= status_code < 500:
            return RetryDecision(retry_on_4xx, 0.0, "invalid_response")
        if status_code >= 500:
            return RetryDecision(
                True,
                exponential_backoff(attempt, base_delay=base_delay, max_delay=max_delay),
                "connectivity_error",
            )

    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return RetryDecision(
            True,
            exponential_backoff(attempt, base_delay=base_delay, max_delay=max_delay),
            "connectivity_error",
        )

    return RetryDecision(
        True,
        exponential_backoff(attempt, base_delay=base_delay, max_delay=max_delay),
        "connectivity_error",
    )


async def run_with_retries(
    attempt_fn: Callable[[int], Awaitable[T]],
    *,
    max_retries: int,
    classify_error: Callable[[Exception, int], RetryDecision],
) -> Tuple[Optional[T], Optional[str], Optional[str]]:
    """Run an async attempt function with centralized retry logic."""
    last_error = None
    last_error_type = None

    for attempt in range(1, max_retries + 1):
        try:
            result = await attempt_fn(attempt)
            return result, None, None
        except RetryableError as exc:
            last_error = str(exc)
            last_error_type = exc.error_type
            if attempt >= max_retries:
                break
            if exc.retry_after:
                await asyncio.sleep(exc.retry_after)
            continue
        except Exception as exc:
            decision = classify_error(exc, attempt)
            last_error = str(exc)
            last_error_type = decision.error_type
            if not decision.retry or attempt >= max_retries:
                break
            if decision.delay:
                await asyncio.sleep(decision.delay)

    return None, last_error or "All retry attempts exhausted", last_error_type
