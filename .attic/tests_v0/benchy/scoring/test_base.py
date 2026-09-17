"""BaseScorer: the ABC every scorer extends.

Contract under test:
- `fitness` defaults to `evaluate(...).value`.
- `aggregate` defaults to a mean of `.value` over the run, keyed `"fitness"`.
- a scorer that doesn't override `__repr__` fails loudly, not silently, so a
  new primitive can never accidentally ship without round-trip support.
- `evaluate` is abstract: BaseScorer itself cannot be instantiated.
"""

from __future__ import annotations

import pytest

from benchy.core import Sample, Score, Scorer
from benchy.scoring.base import BaseScorer


class _Toy(BaseScorer):
    """Minimal concrete scorer: exact numeric equality."""

    name = "toy"

    def evaluate(self, prediction, expected, sample=None):
        return Score(value=1.0 if prediction == expected else 0.0, breakdown={}, scorer=self.name)


class _ToyNoRepr(_Toy):
    pass


class TestBaseScorerIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseScorer()


class TestFitnessDefault:
    def test_fitness_delegates_to_evaluate_value(self):
        toy = _Toy()
        assert toy.fitness(1, 1, None) == 1.0
        assert toy.fitness(1, 2, None) == 0.0

    def test_fitness_accepts_a_sample(self):
        toy = _Toy()
        sample = Sample(id="s1", input={})
        assert toy.fitness(5, 5, sample) == 1.0


class TestAggregateDefault:
    def test_aggregate_of_no_scores_is_zero_fitness(self):
        toy = _Toy()
        agg = toy.aggregate([])
        assert agg["fitness"] == 0.0
        assert agg["n"] == 0

    def test_aggregate_means_the_value_field(self):
        toy = _Toy()
        scores = [Score(value=1.0), Score(value=0.0), Score(value=0.5)]
        agg = toy.aggregate(scores)
        assert agg["fitness"] == pytest.approx(0.5)
        assert agg["n"] == 3


class TestReprMustBeOverridden:
    def test_default_repr_raises_not_implemented(self):
        toy = _ToyNoRepr()
        with pytest.raises(NotImplementedError):
            repr(toy)


class TestSatisfiesCoreProtocol:
    def test_toy_is_a_scorer(self):
        assert isinstance(_Toy(), Scorer)
