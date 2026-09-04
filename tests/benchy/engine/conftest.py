"""Fakes for benchy.benchmark/loss/report/cli tests.

The engine only programs against `benchy.core` protocols (`Task`, `Data`,
`Scorer`, `System`) -- the sibling worktrees building the real
implementations don't exist yet. These fakes are the whole test surface for
this module, which is the point: if the engine's run loop only ever needs
what's declared in core.py, it is genuinely decoupled from its siblings.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from benchy.core import (
    Capabilities,
    Message,
    OntologyPath,
    Prediction,
    Request,
    Response,
    Sample,
    SchemaViolation,
    Score,
)


# --------------------------------------------------------------------------
# FakeTask
# --------------------------------------------------------------------------


@dataclass
class FakeTask:
    """Renders `sample.input['text']` as a user message, parses `response.text` back.

    Stamps `request.meta["sample_id"]` -- pinned by the integration seam
    suite as how a Record correlates back to its sample and how a system
    double induces a per-sample failure.
    """

    name: str = "fake-task"
    ontology: OntologyPath = field(default_factory=lambda: OntologyPath.parse("fake"))
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    invalid_ids: frozenset = field(default_factory=frozenset)
    render_calls: list = field(default_factory=list)
    parse_calls: list = field(default_factory=list)

    def render(self, sample: Sample, caps: Capabilities) -> Request:
        self.render_calls.append(sample.id)
        text = str(sample.input.get("text", sample.id))
        return Request(messages=(Message.text("user", text),), meta={"sample_id": sample.id})

    def parse(self, response: Response, caps: Capabilities) -> Prediction:
        self.parse_calls.append(response)
        return Prediction(value=response.text, raw_text=response.text)

    def validate_sample(self, sample: Sample) -> None:
        if sample.id in self.invalid_ids:
            raise SchemaViolation(f"sample {sample.id!r} fails validation")


# --------------------------------------------------------------------------
# FakeData
# --------------------------------------------------------------------------


@dataclass
class FakeData:
    samples: list

    def __iter__(self):
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def take(self, n: int) -> "FakeData":
        return FakeData(self.samples[:n])

    def split(self, name: str) -> "FakeData":
        return self

    @classmethod
    def of_texts(cls, texts: list[str], expected: list | None = None) -> "FakeData":
        expected = expected if expected is not None else list(texts)
        return cls(
            [
                Sample(id=str(i), input={"text": t}, expected=e)
                for i, (t, e) in enumerate(zip(texts, expected))
            ]
        )


# --------------------------------------------------------------------------
# FakeScorer -- exact match, mean aggregate, always exposes "fitness"
# --------------------------------------------------------------------------


@dataclass
class FakeScorer:
    name: str = "fake_scorer"

    def evaluate(self, prediction: Any, expected: Any, sample: Sample | None) -> Score:
        value = 1.0 if prediction == expected else 0.0
        return Score(value=value, breakdown={"match": value == 1.0}, scorer=self.name)

    def fitness(self, prediction: Any, expected: Any, sample: Sample | None) -> float:
        return self.evaluate(prediction, expected, sample).value

    def aggregate(self, scores):
        scores = list(scores)
        if not scores:
            return {"fitness": 0.0, "mean": 0.0, "n": 0}
        mean = sum(s.value for s in scores) / len(scores)
        return {"fitness": mean, "mean": mean, "n": len(scores)}

    def __repr__(self) -> str:
        return "fake_scorer()"


# --------------------------------------------------------------------------
# FakeSystem + factories for the three required failure shapes
# --------------------------------------------------------------------------


class FakeSystem:
    """A System double whose behavior is fully driven by `respond(request, call_n)`.

    Tracks call count and in-flight concurrency so tests can assert the
    engine respects a concurrency bound. `respond` may return a `Response`
    or raise -- both are meaningful failure channels the engine must handle.
    """

    def __init__(
        self,
        url: str = "fake://system",
        capabilities: Capabilities | None = None,
        respond: Callable[[Request, int], Response] | None = None,
        delay: float | Callable[[Request], float] = 0.0,
    ) -> None:
        self.url = url
        self.capabilities = capabilities or Capabilities(max_concurrency=4)
        self._respond = respond or (lambda req, n: Response(text="ok"))
        self._delay = delay
        self.calls = 0
        self.inflight = 0
        self.max_inflight = 0
        self.closed = False

    def _delay_for(self, request: Request) -> float:
        return self._delay(request) if callable(self._delay) else self._delay

    async def invoke(self, request: Request) -> Response:
        self.calls += 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            delay = self._delay_for(request)
            if delay:
                await asyncio.sleep(delay)
            return self._respond(request, self.calls)
        finally:
            self.inflight -= 1

    async def aclose(self) -> None:
        self.closed = True


def flaky_system(fail_times: int, **kw) -> FakeSystem:
    """Fails with a raised exception `fail_times` times, then succeeds."""
    state = {"n": 0}

    def respond(req: Request, call_n: int) -> Response:
        state["n"] += 1
        if state["n"] <= fail_times:
            raise RuntimeError(f"transient failure #{state['n']}")
        return Response(text="ok")

    return FakeSystem(respond=respond, **kw)


def erroring_system(bad_ids: set[str], **kw) -> FakeSystem:
    """Deterministically returns `Response(error=...)` for samples in `bad_ids`."""

    def respond(req: Request, call_n: int) -> Response:
        sid = req.meta.get("sample_id")
        if sid in bad_ids:
            return Response(error=f"boom on {sid}")
        return Response(text=f"ok:{sid}")

    return FakeSystem(respond=respond, **kw)


def slow_system(delay: float, **kw) -> FakeSystem:
    """Always succeeds, but only after `delay` seconds -- for timeout tests."""
    return FakeSystem(delay=delay, **kw)
