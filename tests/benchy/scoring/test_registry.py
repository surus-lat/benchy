"""registry / register_scorer / parse_scorer.

`parse_scorer` is deliberately a restricted `eval`: a scorer's `repr` is a
call expression naming registered factories with literal arguments, and
`parse_scorer` evaluates that expression against nothing but the registry
and no builtins. This is what "symbolic round-trip" means operationally.
"""

from __future__ import annotations

import pytest

from benchy.core import Sample, Score, Scorer
from benchy.scoring.registry import parse_scorer, register_scorer, registry


class _Dummy:
    """A tiny hand-rolled Scorer used only to test registry mechanics."""

    name = "dummy"

    def __init__(self, k: int = 1):
        self.k = k

    def evaluate(self, prediction, expected, sample=None):
        return Score(value=1.0 if prediction == expected else 0.0)

    def fitness(self, prediction, expected, sample=None):
        return self.evaluate(prediction, expected, sample).value

    def aggregate(self, scores):
        return {"fitness": 0.0}

    def __repr__(self):
        return f"dummy_factory(k={self.k!r})"

    def __eq__(self, other):
        return isinstance(other, _Dummy) and self.k == other.k


def dummy_factory(k: int = 1) -> _Dummy:
    return _Dummy(k=k)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Registering `dummy_factory` under a throwaway name each test avoids
    cross-test collisions with the real registry contents."""
    yield
    registry.pop("dummy_factory", None)
    registry.pop("decorated_factory", None)


class TestRegisterScorerDirectCall:
    def test_registers_under_given_name(self):
        register_scorer("dummy_factory", dummy_factory)
        assert registry["dummy_factory"] is dummy_factory

    def test_returns_the_factory_unchanged(self):
        result = register_scorer("dummy_factory", dummy_factory)
        assert result is dummy_factory

    def test_reregistering_the_same_factory_is_a_no_op(self):
        register_scorer("dummy_factory", dummy_factory)
        register_scorer("dummy_factory", dummy_factory)  # must not raise
        assert registry["dummy_factory"] is dummy_factory

    def test_reregistering_a_different_factory_under_the_same_name_raises(self):
        register_scorer("dummy_factory", dummy_factory)

        def other_factory(k: int = 1) -> _Dummy:
            return _Dummy(k=k)

        with pytest.raises(ValueError):
            register_scorer("dummy_factory", other_factory)


class TestRegisterScorerAsDecorator:
    def test_used_as_a_decorator_with_only_a_name(self):
        @register_scorer("decorated_factory")
        def decorated_factory(k: int = 2) -> _Dummy:
            return _Dummy(k=k)

        assert registry["decorated_factory"] is decorated_factory
        assert decorated_factory().k == 2


class TestParseScorer:
    def setup_method(self):
        register_scorer("dummy_factory", dummy_factory)

    def teardown_method(self):
        registry.pop("dummy_factory", None)

    def test_reconstructs_a_registered_scorer(self):
        original = dummy_factory(k=7)
        rebuilt = parse_scorer(repr(original))
        assert rebuilt == original

    def test_default_kwargs_round_trip_too(self):
        original = dummy_factory()
        rebuilt = parse_scorer(repr(original))
        assert rebuilt == original

    def test_result_satisfies_the_scorer_protocol(self):
        rebuilt = parse_scorer(repr(dummy_factory(k=3)))
        assert isinstance(rebuilt, Scorer)

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_scorer("not_a_registered_scorer(k=1)")

    def test_non_scorer_expression_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_scorer("1 + 1")

    def test_builtins_are_not_available(self):
        with pytest.raises(ValueError):
            parse_scorer("__import__('os').system('true')")

    def test_malformed_expression_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_scorer("dummy_factory(")
