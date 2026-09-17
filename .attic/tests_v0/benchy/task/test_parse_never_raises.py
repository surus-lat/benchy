"""`Task.parse()` must produce a Prediction for every kind of bad Response.

This exercises the full parse path (not just `repair.extract_json`) for
the exact list of malformed-but-recoverable shapes called out in the
build brief: fenced ```json blocks, prose before/after, trailing commas,
single quotes, unescaped newlines in strings, a JSON array when an object
was asked for, an empty string, `None`, and a response with `error` set.
Every one must yield a `Prediction`, never an exception.
"""

from __future__ import annotations

import pytest

from benchy.core import Capabilities, Prediction, Response
from benchy.task import Task

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
}


@pytest.fixture
def task() -> Task:
    return Task(name="x", ontology="x", output=SCHEMA, instructions="Extract name and age.")


CAPS = Capabilities(structured_output=False)

MALFORMED_BUT_RECOVERABLE = [
    ('```json\n{"name": "Ana", "age": 30}\n```', {"name": "Ana", "age": 30}),
    ('Sure, here it is:\n```json\n{"name": "Ana", "age": 30}\n```\nLet me know if you need more.',
     {"name": "Ana", "age": 30}),
    ('Here you go -> {"name": "Ana", "age": 30} <- done.', {"name": "Ana", "age": 30}),
    ('{"name": "Ana", "age": 30,}', {"name": "Ana", "age": 30}),
    ("{'name': 'Ana', 'age': 30}", {"name": "Ana", "age": 30}),
    ('{"name": "Ana\nSmith", "age": 30}', {"name": "Ana\nSmith", "age": 30}),
    ('[{"name": "Ana", "age": 30}]', {"name": "Ana", "age": 30}),
]


class TestMalformedButRecoverable:
    @pytest.mark.parametrize("raw_text,expected_value", MALFORMED_BUT_RECOVERABLE)
    def test_recovers_the_intended_value(self, task, raw_text, expected_value):
        pred = task.parse(Response(text=raw_text), CAPS)
        assert isinstance(pred, Prediction)
        assert pred.parse_ok, pred.parse_error
        assert pred.value == expected_value


class TestUnrecoverableNeverRaises:
    @pytest.mark.parametrize(
        "response",
        [
            Response(text=""),
            Response(text=None),
            Response(text="I'm sorry, I can't help with that."),
            Response(text="{{{{not json at all"),
            Response(error="upstream timeout"),
            Response(error="rate limited", text="please wait"),
        ],
    )
    def test_yields_a_failed_prediction_not_an_exception(self, task, response):
        pred = task.parse(response, CAPS)
        assert isinstance(pred, Prediction)
        assert not pred.parse_ok
        assert isinstance(pred.parse_error, str) and pred.parse_error


class TestEveryModeNeverRaises:
    """Same guarantee across all three render/parse modes."""

    @pytest.mark.parametrize(
        "response",
        [Response(text=""), Response(text=None), Response(error="boom"), Response(text="   ")],
    )
    def test_text_mode(self, response):
        task = Task(name="x", ontology="x", output={"type": "string"}, mode="text")
        pred = task.parse(response, Capabilities())
        assert isinstance(pred, Prediction)

    @pytest.mark.parametrize(
        "response",
        [Response(text=""), Response(text=None), Response(error="boom"), Response(text="not sure at all")],
    )
    def test_choice_mode(self, response):
        task = Task(name="x", ontology="x", mode="choice", labels=["positive", "negative"])
        pred = task.parse(response, Capabilities())
        assert isinstance(pred, Prediction)
        if response.error is not None or not (response.text or "").strip():
            assert not pred.parse_ok
