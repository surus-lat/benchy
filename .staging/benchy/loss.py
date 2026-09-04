"""benchy.loss — export a Benchmark as a loss / metric for Software-3.0 optimizers.

This is the module that makes the vision's headline claim literal: a benchy
Benchmark is not just a report card, it is also *the thing you optimize
against*. `as_loss` is the general case (drive an arbitrary System through
the whole async run loop); `as_metric` is the synchronous, system-free half
of it (just the scoring judgement, for callers -- like a DSPy program's own
forward pass -- that already produced a prediction and only need "is this
good"); `as_dspy_metric` / `as_textgrad_loss` are thin, lazily-imported
adapters onto two popular optimizer frameworks, neither of which is a benchy
dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from benchy.core import LossFn, System

__all__ = ["as_loss", "as_metric", "as_dspy_metric", "as_textgrad_loss"]


def as_loss(benchmark: Any) -> LossFn:
    """`(System) -> Awaitable[float]`. The free variable is the system.

    This is deliberately the whole implementation: `run()` already knows how
    to turn one System into one Report, and `Report.fitness` is already the
    single scalar an optimizer wants. There is nothing else to decide here.
    """

    async def loss(system: System) -> float:
        report = await benchmark.run(system)
        return report.fitness

    return loss


def as_metric(benchmark: Any) -> Callable[[Any, Any, Any], float]:
    """A synchronous per-example metric: `metric(prediction, expected, sample=None) -> float`.

    This is `as_loss` with the System and the async run loop stripped out --
    useful when a prediction already exists (produced by some other means,
    e.g. a DSPy program's own forward pass) and only the scoring judgement is
    needed. It is exactly `benchmark.scoring.fitness`, exposed here so both
    `Benchmark.as_metric()` and `as_dspy_metric()` share one definition.
    """
    scoring = benchmark.scoring

    def metric(prediction: Any, expected: Any, sample: Any = None) -> float:
        return scoring.fitness(prediction, expected, sample)

    return metric


def as_dspy_metric(benchmark: Any) -> Callable[..., float]:
    """Adapt `benchmark.scoring` into a DSPy metric: `metric(example, pred, trace=None) -> float`.

    Lazy import: `dspy` is not a benchy dependency, and this raises a clean,
    actionable `ImportError` if it isn't installed.

    Field convention: `example` is read via `example.expected` (falling back
    to treating `example` itself as the expected value) and, if present,
    `example.sample` (a `benchy.core.Sample`) is threaded through to the
    scorer for rubrics that need the input alongside prediction/expected.
    `pred` is read via `pred.value` (falling back to `pred` itself). A DSPy
    program that wants full control should pass a `Sample` as `example.sample`
    and the raw predicted value as `pred.value`.
    """
    try:
        import dspy  # noqa: F401  (presence check only -- see module docstring)
    except ImportError as exc:
        raise ImportError(
            "as_dspy_metric() requires the 'dspy' package, which is not installed. "
            "Install it with: pip install dspy-ai"
        ) from exc

    metric = as_metric(benchmark)

    def dspy_metric(example: Any, pred: Any, trace: Any = None) -> float:
        expected = getattr(example, "expected", example)
        prediction = getattr(pred, "value", pred)
        sample = getattr(example, "sample", None)
        return metric(prediction, expected, sample)

    return dspy_metric


def as_textgrad_loss(benchmark: Any) -> LossFn:
    """Adapt the benchmark into a TextGrad-shaped loss.

    Lazy import: `textgrad` is not a benchy dependency, and this raises a
    clean, actionable `ImportError` if it isn't installed. TextGrad's own
    losses are typically synchronous callables over a single (input,
    response) pair, but a benchy Benchmark is defined over an entire System
    (many samples); the natural TextGrad-compatible shape for "optimize this
    system end to end" is the same async `(System) -> float` that `as_loss`
    already returns, so a TextGrad-driven optimizer wraps it (e.g. via
    `asyncio.run`) the same way any other loss in that ecosystem gets called
    from a synchronous training loop.
    """
    try:
        import textgrad  # noqa: F401  (presence check only -- see module docstring)
    except ImportError as exc:
        raise ImportError(
            "as_textgrad_loss() requires the 'textgrad' package, which is not installed. "
            "Install it with: pip install textgrad"
        ) from exc

    return as_loss(benchmark)
